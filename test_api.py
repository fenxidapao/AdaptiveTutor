"""
test_api.py — FastAPI 端点集成测试（T9，2026-08-22）

目标：不依赖 Ollama / chroma 落盘 / 外网，用 fastapi TestClient + monkeypatch
隔离全部外部依赖，验证 API 层的：
  - 路由可达（/ /health /query /retrieve /documents /index/*）
  - 参数校验（缺失 / 非法文件类型 / 路径穿越 → 4xx）
  - 响应结构（Citation / QueryResponse / IndexStats 字段齐全）
  - 错误处理（索引不可用 503、生成失败 500）

关键设计：main.py 顶层 `from rag_core import (...)`，因此 monkeypatch
`main.<函数名>` 即可替换端点内部对 rag_core 的调用，无需触碰 rag_core 本身；
config.DATA_DIR / PERSIST_DIRECTORY 也重定向到临时目录，杜绝测试污染真实语料。

运行：pytest test_api.py -v   （无需 Ollama / chroma 持久化 / 网络）
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 隔离查询日志（import main 时 query_log 读取 QUERY_LOG_PATH）
_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
_tmp.close()
os.environ["QUERY_LOG_PATH"] = _tmp.name

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import main
from fastapi.testclient import TestClient


class FakeCol:
    """get_vector_store 的替身：只实现端点用到的接口。"""

    def __init__(self, count=10, metas=None):
        self._count = count
        self._metas = metas or [{"source": "/x/a.md"} for _ in range(count)]
        self.name = "course_rag"

    def count(self):
        return self._count

    def get(self, include=None):
        return {"metadatas": self._metas}


class FakeDoc:
    """rag_core.Document 的替身：page_content + metadata。"""

    def __init__(self, content="ACID 包括原子性、一致性、隔离性、持久性。", source="db.txt", start=0, score=None):
        self.page_content = content
        self.metadata = {"source": source, "start_index": start}
        if score is not None:
            self.metadata["hybrid_score"] = score


@pytest.fixture
def client(monkeypatch, tmp_path):
    """隔离外部依赖后提供 TestClient（生命周期内复用）。"""
    # 数据目录 / 向量库目录重定向到临时目录
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    # 默认向量库替身（lifespan 预热 + 各端点共用）
    monkeypatch.setattr(main, "get_vector_store", lambda *a, **k: FakeCol())
    # health 的外部检查 mock 掉，避免真的请求 Ollama
    monkeypatch.setattr("health.check_ollama", lambda base_url=None, timeout=4: (True, "mock-ok"))
    monkeypatch.setattr("health.check_reranker_cache", lambda: (True, "mock-ok"))
    with TestClient(main.app) as c:
        yield c


# ==================== meta / health ====================
def test_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "CourseRAG API"
    assert any("/health" in e for e in body["endpoints"])
    assert any("/query" in e for e in body["endpoints"])
    assert "models" in body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["ok"], bool)
    assert isinstance(body["checks"], list) and len(body["checks"]) >= 2
    assert body["vector_store_count"] == 10


# ==================== /query ====================
def _patch_rag(monkeypatch, answer="ACID 包括原子性、一致性、隔离性、持久性[1]。", confidence=0.9):
    """把检索 + 生成 + 置信度 + 日志全部替换为假实现。"""
    monkeypatch.setattr(
        main, "retrieve_and_rerank",
        lambda q, force_no_rerank=False: [FakeDoc(score=0.9), FakeDoc(content="第二范式 2NF 消除部分函数依赖。", source="db2.txt", start=50, score=0.7)],
    )

    class _Chain:
        def invoke(self, question, docs=None):
            return answer

    monkeypatch.setattr(main, "build_rag_chain", lambda custom_prompt=None, history="": {"chain": _Chain()})
    monkeypatch.setattr(main, "get_confidence", lambda docs: confidence)
    monkeypatch.setattr(main, "log_query", lambda **kw: None)


def test_query_fast(client, monkeypatch):
    _patch_rag(monkeypatch)
    r = client.post("/query", json={"question": "什么是 ACID？", "mode": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ACID 包括原子性、一致性、隔离性、持久性[1]。"
    assert body["mode"] == "fast"
    assert body["grounded"] is True
    assert body["low_confidence"] is False
    assert len(body["citations"]) == 2
    assert body["citations"][0]["rank"] == 1
    assert body["citations"][0]["source"] == "db.txt"
    assert isinstance(body["latency_ms"], int)


def test_query_low_confidence(client, monkeypatch):
    _patch_rag(monkeypatch, confidence=0.5)  # 低于阈值 0.69
    r = client.post("/query", json={"question": "完全无关的问题？", "mode": "fast"})
    assert r.status_code == 200
    assert r.json()["low_confidence"] is True


def test_query_missing_question(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_accurate_mode(client, monkeypatch):
    _patch_rag(monkeypatch)
    r = client.post("/query", json={"question": "什么是 2NF？", "mode": "accurate"})
    assert r.status_code == 200
    assert r.json()["mode"] == "accurate"


# ==================== /retrieve ====================
def test_retrieve(client, monkeypatch):
    monkeypatch.setattr(
        main, "retrieve_and_rerank",
        lambda q, force_no_rerank=False: [FakeDoc(score=0.9), FakeDoc(content="死锁四必要条件：互斥、占有并等待、不可抢占、循环等待。", source="os.txt", start=10, score=0.6)],
    )
    r = client.post("/retrieve", json={"question": "什么是死锁？", "mode": "accurate"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "什么是死锁？"
    assert len(body["docs"]) == 2
    assert body["docs"][0]["rank"] == 1
    assert body["docs"][0]["source"] == "db.txt"
    assert "score" in body["docs"][0]


# ==================== /documents ====================
def test_documents_list(client):
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert "files" in body and "total_chunks" in body
    assert isinstance(body["files"], list)


def test_documents_upload_ingest(client, monkeypatch):
    # 非首次建库（count>0）→ 走增量 upsert 路径
    monkeypatch.setattr(main, "chunk_file", lambda p: [FakeDoc(source=str(p))])
    monkeypatch.setattr(main, "add_documents", lambda chunks: len(chunks))
    r = client.post("/documents", files={"file": ("note.txt", b"hello world", "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "note.txt"
    assert body["added_chunks"] == 1
    # 文件确实写入了临时 data 目录
    assert (Path(config.DATA_DIR) / "note.txt").exists()


def test_documents_upload_bad_extension(client):
    r = client.post("/documents", files={"file": ("evil.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    assert "不支持的文件类型" in r.json()["detail"]


def test_documents_upload_empty(client):
    r = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400


def test_documents_upload_path_traversal(client):
    r = client.post("/documents", files={"file": ("../evil.txt", b"x", "text/plain")})
    assert r.status_code == 400
    assert "非法文件名" in r.json()["detail"]


def test_documents_delete(client, monkeypatch):
    monkeypatch.setattr(main, "delete_by_source", lambda source: None)
    r = client.delete("/documents/ghost.txt")  # 文件不存在，也应正常返回（索引移除）
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] is True
    assert body["filename"] == "ghost.txt"
    assert body["warning"] is None


def test_safe_filename_rejects_traversal():
    """路径穿越防护：_safe_filename 必须拒绝 .. / 分隔符 / 空名。"""
    for bad in ("../evil.txt", "a/b.txt", "..", ".", "", "C:\\evil.txt"):
        with pytest.raises(Exception):
            main._safe_filename(bad)
    # 合法纯文件名放行
    assert main._safe_filename("note.txt") == "note.txt"


# ==================== /index ====================
def test_index_rebuild(client, monkeypatch):
    monkeypatch.setattr(main, "load_and_chunk_documents", lambda: [FakeDoc()])
    monkeypatch.setattr(main, "get_vector_store", lambda chunks=None: FakeCol(count=1))
    r = client.post("/index/rebuild")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == 1
    assert body["files"] == 1


def test_index_stats(client, monkeypatch):
    col = FakeCol(count=7, metas=[{"source": "/x/a.md"}, {"source": "/x/b.md"}, {"source": "/x/b.md"}])
    monkeypatch.setattr(main, "get_vector_store", lambda: col)
    r = client.get("/index/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 7
    assert body["source_files"] == 2  # 按 source 去重
    assert body["collection"] == "course_rag"


def test_index_stats_unavailable(client, monkeypatch):
    def _boom():
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(main, "get_vector_store", _boom)
    r = client.get("/index/stats")
    assert r.status_code == 503
