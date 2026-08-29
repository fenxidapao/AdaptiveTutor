# -*- coding: utf-8 -*-
"""CourseRAG MCP stdio server（服务端零依赖：仅 Python 标准库）

MCP 协议最小子集（stdio 传输）：
  - 帧：stdin/stdout 上按行分隔的 JSON-RPC 2.0 消息（UTF-8，单行内不得含换行）
  - 方法：initialize / notifications/initialized / ping / tools/list / tools/call
  - 能力：仅声明 tools（未声明 resources/prompts，客户端不应调用对应方法）
  - 工具：course_retrieve——与 FastAPI /retrieve 同一检索实现（retrieve_and_rerank），
    mode 语义一致（fast=纯向量秒级 / accurate=混合检索+reranker 精排），默认 fast。

硬约束：
  - stdout 是协议通道，本文件任何日志/调试输出只能走 stderr；
  - rag_core/chromadb 等第三方代码可能 print 到 stdout，其 import、建库、检索
    全程必须在 stdout 重定向保护下运行，否则会污染 JSON-RPC 流。
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

SUPPORTED_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}
FALLBACK_VERSION = "2024-11-05"
SERVER_INFO = {"name": "courserag-mcp", "version": "0.1.0"}

TOOLS = [
    {
        "name": "course_retrieve",
        "description": (
            "检索本地课程知识库（计算机/数学专业课笔记、教材、合规公开教程），"
            "返回带来源与相关性分数的文本块，作为课程知识点问答的事实依据。"
            "mode=fast 纯向量召回（秒级）；mode=accurate 向量+BM25 混合检索并经 "
            "bge-reranker 精排（质量更高，CPU 上可能需要数十秒）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "课程/课本知识点问题或检索词"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5,
                          "description": "返回的文本块数量"},
                "mode": {"type": "string", "enum": ["fast", "accurate"], "default": "fast",
                         "description": "检索模式，语义与 CourseRAG API /retrieve 一致"},
            },
            "required": ["query"],
        },
    }
]

_RAG = {"ready": False}


class JsonRpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class _StdoutToFd2:
    """OS 级 fd1 保护：rag 工作期间把真实 fd1 指向 stderr。
    contextlib.redirect_stdout 只能拦 Python 层 print；tqdm/torch 等可能绕过
    sys.stdout 直写 fd1（实测 reranker「Loading weights」进度条污染过协议流），
    必须在文件描述符层重定向才彻底。退出时恢复。"""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        os.dup2(2, 1)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)


def _log(msg: str) -> None:
    print(f"[courserag-mcp] {msg}", file=sys.stderr, flush=True)


def _ensure_rag() -> None:
    """懒加载 rag_core 并预热向量库（与 main.py lifespan 同一 get_vector_store 语义，
    自动复用 chroma_db 现有索引）。stdout 全程重定向到 stderr。"""
    if _RAG["ready"]:
        return
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr), _StdoutToFd2():
        import rag_core
        rag_core.get_vector_store()
    _RAG["ready"] = True
    _log(f"rag_core ready in {time.time() - t0:.1f}s")


def _retrieve_real(query: str, top_k: int, mode: str) -> str:
    _ensure_rag()
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr), _StdoutToFd2():
        from rag_core import retrieve_and_rerank
        docs = retrieve_and_rerank(query, force_no_rerank=(mode != "accurate"))
    docs = docs[:top_k]
    _log(f"retrieve done in {time.time() - t0:.1f}s (mode={mode}, hits={len(docs)})")
    if not docs:
        return "未检索到相关内容（知识库无匹配文本块）。"
    lines = [f"检索到 {len(docs)} 个相关文本块（mode={mode}）：", ""]
    for rank, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        score = doc.metadata.get("rerank_score") or doc.metadata.get("hybrid_score")
        start = doc.metadata.get("start_index")
        meta = f"来源: {source}"
        if score is not None:
            meta += f" | 相关性分数: {float(score):.3f}"
        if start is not None:
            meta += f" | 原文偏移: {start}"
        lines.append(f"[{rank}] {meta}")
        lines.append(doc.page_content.strip())
        lines.append("")
    return "\n".join(lines)


def _call_tool(name: str, arguments: dict) -> dict:
    if name != "course_retrieve":
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise JsonRpcError(-32602, "invalid params: 'query' (string) is required")
    try:
        top_k = int(arguments.get("top_k", 5))
    except (TypeError, ValueError):
        raise JsonRpcError(-32602, "invalid params: 'top_k' must be an integer")
    mode = arguments.get("mode", "fast")
    if mode not in ("fast", "accurate"):
        raise JsonRpcError(-32602, "invalid params: 'mode' must be 'fast' or 'accurate'")
    try:
        text = _retrieve_real(query.strip(), top_k, mode)
    except Exception as exc:  # 检索失败降级为 isError 结果，不让进程退出
        _log(f"retrieve failed: {exc!r}")
        text = (
            f"检索失败：{exc}\n"
            "请确认 ollama serve 已启动（嵌入模型 nomic-embed-text 可用）"
            "且 chroma_db 索引存在。"
        )
        return {"content": [{"type": "text", "text": text}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _dispatch(method: str, params: dict):
    if method == "initialize":
        # 版本协商：支持则回显客户端版本，否则回退到本服务端最新支持版
        requested = (params or {}).get("protocolVersion")
        version = requested if requested in SUPPORTED_VERSIONS else FALLBACK_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = params or {}
        return _call_tool(params.get("name", ""), params.get("arguments") or {})
    if method == "ping":
        return {}
    raise JsonRpcError(-32601, f"Method not found: {method}")


def _send(payload: dict) -> None:
    # ensure_ascii=True：线上帧纯 ASCII（非 ASCII 一律 \uXXXX 转义）。
    # stdio 在 Windows 上会跨代码页边界，GBK 误解码对 ASCII 是恒等变换——
    # 转义帧从数学上杜绝中转层乱码；任何合规 JSON 客户端都会解回原文。
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=True).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    _log(f"start (stdio; supported versions: {', '.join(sorted(SUPPORTED_VERSIONS))})")
    for raw in sys.stdin.buffer:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _send({"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700, "message": f"Parse error: {exc}"}})
            continue
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if msg_id is None:  # 通知（无 id）不回包
            _log(f"notification: {method}")
            continue
        try:
            result = _dispatch(method, msg.get("params") or {})
            _send({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except JsonRpcError as exc:
            _send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": exc.code, "message": exc.message}})
        except Exception as exc:  # 协议层兜底：任何单条异常都不能带崩整条流
            _log(f"internal error on {method}: {exc!r}")
            _send({"jsonrpc": "2.0", "id": msg_id,
                   "error": {"code": -32603, "message": f"Internal error: {exc}"}})
    _log("stdin closed, exit")


if __name__ == "__main__":
    main()
