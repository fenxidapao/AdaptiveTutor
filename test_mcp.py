"""
test_mcp.py — MCP stdio 服务端测试（2026-08-29，配套 mcp_server.py）

两组用例：
  - 协议组（零依赖，CI 可跑）：子进程管道直测 mcp_server.py 的 JSON-RPC 行为
    （版本协商 / 通知不回包 / tools list schema / 错误码 / 线上帧纯 ASCII）
    + 进程内单测参数钳制（top_k / mode / query 校验——import mcp_server 不触发
    rag_core 懒加载，故零 chromadb/ollama 依赖）
  - 冒烟组（需 Ollama + 索引，无 Ollama 自动 skip）：tools/call 端到端真实检索

与官方 MCP Inspector 互操作证据见 mcp_interop_evidence.md（本文件是可回归的
自动化保护层，互操作实测是一次性证据，两者互补）。
"""
import json
import os
import subprocess
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import mcp_server

MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


# ==================== 子进程管道助手 ====================
class McpProc:
    """起一个 mcp_server.py 子进程，按行收发 JSON-RPC（线上帧应为纯 ASCII）。"""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, MCP_SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.wire = b""  # 累计服务端线上字节，供纯 ASCII 断言

    def send_raw(self, data: bytes):
        self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()

    def send(self, payload: dict):
        self.send_raw(json.dumps(payload, ensure_ascii=True).encode("ascii"))

    def recv(self):
        line = self.proc.stdout.readline()
        assert line, "服务端在收到响应前退出（stdin 关闭或崩溃）"
        self.wire += line
        return json.loads(line.decode("ascii"))

    def handshake(self, version="2025-06-18"):
        self.send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": version, "capabilities": {},
                              "clientInfo": {"name": "test", "version": "0"}}})
        resp = self.recv()
        assert resp["id"] == 1 and "result" in resp
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


@pytest.fixture()
def mcp():
    p = McpProc()
    yield p
    p.close()


# ==================== 协议组（零依赖，CI 可跑） ====================
def test_initialize_echoes_supported_version(mcp):
    result = mcp.handshake("2025-06-18")
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"]
    assert "tools" in result["capabilities"]


def test_initialize_falls_back_for_unknown_version(mcp):
    result = mcp.handshake("1999-01-01")
    assert result["protocolVersion"] in mcp_server.SUPPORTED_VERSIONS


def test_notification_gets_no_reply(mcp):
    mcp.handshake()
    # 唯一在途回包应属于 ping(id=2)；若通知被错误回包，这条会先收到别的东西
    mcp.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    resp = mcp.recv()
    assert resp["id"] == 2 and resp["result"] == {}


def test_tools_list_schema(mcp):
    mcp.handshake()
    mcp.send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    tools = mcp.recv()["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "course_retrieve" in names
    schema = next(t for t in tools if t["name"] == "course_retrieve")["inputSchema"]
    assert "query" in schema["required"]
    assert "query" in schema["properties"]
    assert schema["properties"]["mode"]["enum"] == ["fast", "accurate"]


def test_unknown_method_returns_32601(mcp):
    mcp.handshake()
    mcp.send({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
    resp = mcp.recv()
    assert resp["error"]["code"] == -32601


def test_malformed_json_returns_32700(mcp):
    mcp.send_raw(b"{broken json")
    resp = mcp.recv()
    assert resp["error"]["code"] == -32700


def test_wire_is_pure_ascii(mcp):
    mcp.handshake()
    mcp.send({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
    mcp.recv()
    assert all(b < 128 for b in mcp.wire), "协议帧混入非 ASCII 字节（Windows 跨代码页会乱码）"


def test_eof_exits_cleanly(mcp):
    mcp.handshake()
    mcp.close()  # close() 内 wait 超时即失败
    assert mcp.proc.returncode == 0


# ==================== 参数钳制（进程内单测，零依赖） ====================
def _with_stub_retrieve(monkeypatch, calls):
    def fake_retrieve(query, top_k, mode):
        calls.append((query, top_k, mode))
        return "stub 检索结果"
    monkeypatch.setattr(mcp_server, "_retrieve_real", fake_retrieve)


def test_top_k_clamped_to_schema_range(monkeypatch):
    calls = []
    _with_stub_retrieve(monkeypatch, calls)
    mcp_server._call_tool("course_retrieve", {"query": "ACID", "top_k": -5})
    mcp_server._call_tool("course_retrieve", {"query": "ACID", "top_k": 999})
    mcp_server._call_tool("course_retrieve", {"query": "ACID"})  # 缺省 5
    assert [c[1] for c in calls] == [1, 20, 5]


def test_invalid_params_rejected(monkeypatch):
    calls = []
    _with_stub_retrieve(monkeypatch, calls)
    with pytest.raises(mcp_server.JsonRpcError) as e:
        mcp_server._call_tool("course_retrieve", {"query": "  "})
    assert e.value.code == -32602
    with pytest.raises(mcp_server.JsonRpcError) as e:
        mcp_server._call_tool("course_retrieve", {"query": "ACID", "mode": "bogus"})
    assert e.value.code == -32602
    result = mcp_server._call_tool("no_such_tool", {})
    assert result["isError"] is True
    assert calls == []  # 非法参数不应触达检索


def test_retrieve_failure_degrades_to_is_error(monkeypatch):
    def boom(query, top_k, mode):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(mcp_server, "_retrieve_real", boom)
    result = mcp_server._call_tool("course_retrieve", {"query": "ACID"})
    assert result["isError"] is True
    assert "ollama" in result["content"][0]["text"].lower() or "失败" in result["content"][0]["text"]


# ==================== 冒烟组（需 Ollama + 索引） ====================
def _ollama_ready():
    try:
        base = config.OLLAMA_BASE_URL.rstrip("/")
        with urllib.request.urlopen(base + "/api/tags", timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama 未启动，跳过 MCP 真实检索冒烟")
def test_tools_call_real_retrieve_end_to_end():
    mcp = McpProc()
    try:
        mcp.handshake()
        mcp.send({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                  "params": {"name": "course_retrieve",
                             "arguments": {"query": "什么是ACID", "top_k": 2, "mode": "fast"}}})
        result = mcp.recv()["result"]
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "来源" in text and "ACID" in text
    finally:
        mcp.close()
