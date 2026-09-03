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
def _patch_answer(monkeypatch, answer="ACID 包括原子性、一致性、隔离性、持久性[1]。", low_confidence=False):
    """把统一问答入口 answer_question 替换为假实现（评审 P1 起 /query 走该入口，
    原先 mock 的 retrieve_and_rerank / build_rag_chain / get_confidence 已不在
    API 层直接调用）。"""
    citations = [
        {"rank": 1, "source": "db.txt", "content": "ACID 包括原子性、一致性、隔离性、持久性。",
         "score": 0.9, "start_index": 0},
        {"rank": 2, "source": "db2.txt", "content": "第二范式 2NF 消除部分函数依赖。",
         "score": 0.7, "start_index": 50},
    ]

    def _fake_answer_question(question, mode="fast", **kwargs):
        return {
            "answer": answer,
            "citations": citations,
            "docs": None,
            "cached": False,
            "cache_sim": None,
            "rejected": False,
            "reject_signal": None,
            "grounding_ran": mode == "accurate",
            "grounded": True,
            "not_grounded": False,
            "grounded_score": 1.0 if mode == "accurate" else None,
            "low_confidence": low_confidence,
        }

    monkeypatch.setattr(main, "answer_question", _fake_answer_question)
    monkeypatch.setattr(main, "log_query", lambda **kw: None)


def test_query_fast(client, monkeypatch):
    _patch_answer(monkeypatch)
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
    _patch_answer(monkeypatch, low_confidence=True)  # 低于阈值 0.69
    r = client.post("/query", json={"question": "完全无关的问题？", "mode": "fast"})
    assert r.status_code == 200
    assert r.json()["low_confidence"] is True


def test_query_missing_question(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_accurate_mode(client, monkeypatch):
    _patch_answer(monkeypatch)
    r = client.post("/query", json={"question": "什么是 2NF？", "mode": "accurate"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "accurate"
    # 评审 P1 后 accurate 在 API 侧同样过闸门：正常路径 grounded/cached 字段语义不变
    assert body["grounded"] is True
    assert body["cached"] is False


# ==================== rag_core.answer_question 统一闸门（评审 P1） ====================
def _patch_answer_pipeline(monkeypatch, docs, gen_answer="好的回答[1]。", grounded=(True, 1.0)):
    """隔离 answer_question 的外部依赖：检索 / 链 / 接地自检 / 语义缓存。
    返回 store_calls 列表，用于断言缓存写入的 reject 标志。"""
    import rag_core

    monkeypatch.setattr(rag_core, "retrieve_and_rerank", lambda q, force_no_rerank=False: docs)

    class _Chain:
        def invoke(self, question, docs=None):
            return gen_answer

    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _Chain()})
    monkeypatch.setattr(rag_core, "grounded_generate",
                        lambda q, d, chain: (gen_answer, grounded[0], grounded[1]))
    store_calls = []
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: store_calls.append(reject))
    return store_calls


def _mk_doc(vector_score):
    import rag_core

    d = rag_core.Document("ACID 包括原子性。", {"source": "/x/db.txt", "start_index": 0})
    d.metadata["vector_score"] = vector_score
    return d


def test_answer_question_fast_no_hard_reject(monkeypatch):
    """快速模式：即使 top-1 向量分很低也不硬拒（只软提示 low_confidence）。"""
    import rag_core

    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.30)])
    # use_cache=True + lookup mock 未命中：验证「未命中→检索→写入」完整缓存路径
    res = rag_core.answer_question("什么是ACID", mode="fast", use_cache=True)
    assert res["rejected"] is False
    assert res["low_confidence"] is True  # 0.30 < 0.69 软提示
    assert res["answer"] == "好的回答[1]。"
    assert res["grounding_ran"] is False  # 快速模式不跑接地自检


def test_answer_question_accurate_hard_reject(monkeypatch):
    """精确模式：top-1 向量分低于 HARD_REJECT_VECTOR_THRESHOLD → 硬拒 + 不入缓存。"""
    import rag_core

    store_calls = _patch_answer_pipeline(monkeypatch, [_mk_doc(0.30)])
    res = rag_core.answer_question("怎么追女生", mode="accurate", use_cache=True)
    assert res["rejected"] is True
    assert res["answer"] == rag_core.HARD_REJECT_ANSWER
    assert res["not_grounded"] is False  # 检索拒而非接地拒
    assert store_calls == [True]  # 拒答不缓存


def test_answer_question_accurate_normal(monkeypatch):
    """精确模式正常路径：过闸门 → 接地自检通过 → 入缓存（reject=False）。"""
    import rag_core

    store_calls = _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)], grounded=(True, 0.95))
    res = rag_core.answer_question("什么是ACID", mode="accurate", use_cache=True)
    assert res["rejected"] is False
    assert res["not_grounded"] is False
    assert res["grounded_score"] == 0.95
    assert res["answer"] == "好的回答[1]。"
    assert store_calls == [False]


def test_answer_question_fast_stream(monkeypatch):
    """评审 P3 真实流式：快速模式 on_token 逐段回调，聚合回完整答案
    （缓存/日志语义不变——store 收到的仍是完整答案而非片段）。"""
    import rag_core

    pieces = ["ACID ", "包括", "原子性[1]。"]

    class _StreamChain:
        def invoke_stream(self, question, docs=None):
            yield from pieces

    monkeypatch.setattr(rag_core, "retrieve_and_rerank",
                        lambda q, force_no_rerank=False: [_mk_doc(0.90)])
    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _StreamChain()})
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    store_calls = []
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: store_calls.append((a, reject)))
    got = []
    res = rag_core.answer_question("什么是ACID", mode="fast",
                                   on_token=got.append, use_cache=True)
    assert got == pieces  # 生成中逐段回调
    assert res["answer"] == "ACID 包括原子性[1]。"  # 聚合为完整答案
    assert store_calls == [("ACID 包括原子性[1]。", False)]


def test_answer_question_accurate_stream_ignored(monkeypatch):
    """评审 P3 安全语义：精确模式即使传 on_token 也不流式（接地自检需完整答案，
    未接地时整段替换为拒答文案——边生成边展示会让已展示内容无法撤回）。"""
    import rag_core

    class _Chain:
        def invoke(self, question, docs=None):
            return "缓冲路径的完整答案[1]。"

    monkeypatch.setattr(rag_core, "retrieve_and_rerank",
                        lambda q, force_no_rerank=False: [_mk_doc(0.90)])
    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _Chain()})
    monkeypatch.setattr(rag_core, "grounded_generate",
                        lambda q, d, chain: ("缓冲路径的完整答案[1]。", True, 1.0))
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: None)
    got = []
    res = rag_core.answer_question("什么是ACID", mode="accurate",
                                   on_token=got.append, use_cache=True)
    assert got == []  # 未触发任何流式回调
    assert res["answer"] == "缓冲路径的完整答案[1]。"


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
    """路径穿越防护：_safe_filename 必须拒绝 .. / 分隔符 / 空名。

    注意：路径穿越样例用 os.path.sep / os.path.altsep 跨平台构造——
    写死 'C:\\evil.txt' 只在 Windows 上是穿越（反斜杠是分隔符），Linux 上
    反斜杠是普通字符、Path.name 等于原串导致误判（CI 曾在此挂，2026-08-22）。
    """
    bad = ["..", ".", "", ".." + os.path.sep + "evil.txt"]
    if os.path.altsep:  # Windows 的 altsep 是 '/'; POSIX 为 None
        bad.append(".." + os.path.altsep + "evil.txt")
    for name in bad:
        with pytest.raises(Exception):
            main._safe_filename(name)
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


def test_index_rebuild_empty_data(client, monkeypatch):
    """空知识库重建（P1 回归，2026-08-22）：data/ 无文档时 POST /index/rebuild
    不应 500——get_vector_store([]) 对 chromadb 空 upsert 曾抛
    'Non-empty lists are required'，已修复为返回空集合。"""
    monkeypatch.setattr(main, "load_and_chunk_documents", lambda: [])
    r = client.post("/index/rebuild")
    # main.rebuild_index 对空 chunks 显式返回 400（有提示），而不是 500 崩溃
    assert r.status_code == 400
    assert "没有可索引的文档" in r.json()["detail"]


def test_get_vector_store_empty_chunks(monkeypatch, tmp_path):
    """P1 回归（2026-08-22）：rag_core.get_vector_store([]) 不应崩溃。

    修复前：chromadb 对空 upsert 抛 ValueError('Non-empty lists are required')，
    空 data/ 目录点「全量重建」直接崩。修复后：跳过 upsert、返回 count()==0 的空集合。
    本用例隔离持久化目录，不依赖 Ollama（空 chunks 不触发嵌入）。
    """
    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    import rag_core

    # 泄漏防护（与下方 test_get_vector_store_readonly_no_rebuild 同款）：
    # get_vector_store 内部会把 _chroma_client/_vector_store 模块级单例绑定到本测试的
    # 临时目录；不重置的话 monkeypatch 恢复 config 后单例仍指向 tmp，后续 test_retrieval
    # 全量跑会连到空库（Ollama 未启动时该组被 skip，泄漏不可见——2026-09-03 暴露修复）
    monkeypatch.setattr(rag_core, "_chroma_client", None)
    monkeypatch.setattr(rag_core, "_vector_store", None)
    col = rag_core.get_vector_store([])
    assert col.count() == 0


def test_index_stats(client, monkeypatch):
    col = FakeCol(count=7, metas=[{"source": "/x/a.md"}, {"source": "/x/b.md"}, {"source": "/x/b.md"}])
    # 评审 P2-3 后 /index/stats 走 get_vector_store(auto_rebuild=False)，mock 须收 kwargs
    monkeypatch.setattr(main, "get_vector_store", lambda *a, **k: col)
    r = client.get("/index/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 7
    assert body["source_files"] == 2  # 按 source 去重
    assert body["collection"] == "course_rag"


def test_index_stats_unavailable(client, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(main, "get_vector_store", _boom)
    r = client.get("/index/stats")
    assert r.status_code == 503


# ==================== 鉴权 / 上传上限（评审 P3 安全小项） ====================
def test_require_auth_enforced_when_enabled(client, monkeypatch):
    """AUTH_ENABLED=true：无密钥/错密钥 401，正确密钥放行。
    （常量时间比较 secrets.compare_digest 只改实现不改行为——防时序侧信道。）"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "API_KEY", "secret-key")
    r = client.get("/health")
    assert r.status_code == 401
    r = client.get("/health", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401
    r = client.get("/health", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_documents_upload_too_large(client, monkeypatch):
    """评审 P3 回归：超过 MAX_UPLOAD_MB → 413，且有界读取不落盘。"""
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)  # 上限置 0：任何非空内容都超限
    r = client.post("/documents", files={"file": ("big.txt", b"x" * 10, "text/plain")})
    assert r.status_code == 413
    assert not (Path(config.DATA_DIR) / "big.txt").exists()


# ==================== 评审 P2 回归 ====================
def test_split_structured_offset_remap():
    """评审 P2-1 回归：含代码块的文档，切分产出的 start_index 必须是「原文偏移」。
    修复前偏移落在哨兵保护后的坐标系——代码块（原文 314 字符）被换成 4 字符哨兵后，
    其后所有块的 start_index 整体前漂 310 字符，text[start:start+len(chunk)] != chunk。"""
    import rag_core

    head = "前导文字。" * 30                            # 150 字符单行
    code = "```python\n" + "x = 1\n" * 50 + "```"      # 314 字符 → 哨兵 §B0§ 仅 4 字符
    text = head + "\n" + code + "\n" + "尾注文字。" * 30  # 尾注起点原文偏移 = 466
    chunks = rag_core._split_structured(text, 200, 0, rag_core._SEPARATORS)
    assert len(chunks) >= 2
    hit_code = False
    for chunk, start, _chapter in chunks:
        assert 0 <= start and start + len(chunk) <= len(text)
        # 核心断言：报告的偏移处，原文必须与块内容逐字对齐（引用溯源可定位的前提）
        assert text[start : start + len(chunk)] == chunk, (
            f"start_index={start} 处原文与块内容不匹配（偏移漂移）"
        )
        if "```" in chunk:
            hit_code = True
    assert hit_code  # 代码块被整体保护并进入某个块


def test_bm25_cache_built_once_per_corpus(monkeypatch):
    """评审 P2-4 回归：BM25 索引按语料版本缓存（同语料两次查询只建一次），
    _invalidate_all_chunks（增删/重建索引）后重建。"""
    import rag_core

    docs = [rag_core.Document("事务的 ACID 特性。", {"source": "a.txt", "start_index": 0}),
            rag_core.Document("死锁的四个必要条件。", {"source": "a.txt", "start_index": 20})]

    class _FakeCol:
        def count(self):
            return len(docs)

        def get(self, include=None):
            return {"documents": [d.page_content for d in docs],
                    "metadatas": [d.metadata for d in docs]}

    monkeypatch.setattr(rag_core, "get_vector_store", lambda *a, **k: _FakeCol())
    monkeypatch.setattr(rag_core, "retrieve_vector",
                        lambda q, k=None: [rag_core.Document("x", {"source": "a.txt", "start_index": 0})])
    built = {"n": 0}
    _orig_bm25 = rag_core._BM25

    def _spy(d):
        built["n"] += 1
        return _orig_bm25(d)

    monkeypatch.setattr(rag_core, "_BM25", _spy)
    monkeypatch.setattr(rag_core, "_all_chunks_cache", None)
    monkeypatch.setattr(rag_core, "_all_chunks_count", -1)
    monkeypatch.setattr(rag_core, "_bm25_cache", None)

    rag_core.hybrid_retrieve("ACID", k=5)
    rag_core.hybrid_retrieve("ACID", k=5)
    assert built["n"] == 1, "同语料两次查询不应重建 BM25 索引"
    rag_core._invalidate_all_chunks()
    rag_core.hybrid_retrieve("ACID", k=5)
    assert built["n"] == 2, "语料失效（增删/重建）后应重建 BM25 索引"


def test_get_vector_store_readonly_no_rebuild(monkeypatch, tmp_path):
    """评审 P2-3 回归：auto_rebuild=False 时空库不触发全量重建（只读端点零副作用）。"""
    import rag_core

    def _no_rebuild():
        raise AssertionError("auto_rebuild=False 不应触发 load_and_chunk_documents")

    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_core, "load_and_chunk_documents", _no_rebuild)
    # 绕开其他用例遗留的模块级单例缓存，确保真实走「空库 + auto_rebuild=False」分支
    monkeypatch.setattr(rag_core, "_chroma_client", None)
    monkeypatch.setattr(rag_core, "_vector_store", None)
    col = rag_core.get_vector_store(auto_rebuild=False)
    assert col.count() == 0  # 空库如实返回，没有建索引副作用
