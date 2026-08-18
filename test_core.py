"""
test_core.py — CourseRAG 核心函数冒烟 / 单测（P2-1）

设计目标：可在「无 LLM 栈」环境直接运行（CI / 沙箱 / 本机 venv 外），
不依赖 chromadb / ollama / sentence_transformers / streamlit / langchain：
  - 检索算法来自 rag_search_utils（仅依赖标准库 re / math / hashlib）
  - 查询日志来自 query_log（仅依赖标准库）
  - 文档对象用轻量 duck-typed 替身 Doc，只需 page_content + metadata 两属性

覆盖的「核心函数」：
  rag_search_utils: _tokenize / _BM25 / _bm25_rank / _rrf_fuse / _stable_ids
  query_log:        _is_grounded / log_query（落盘格式）

（P1-2 的 render_answer_html / build_history_text 在 app.py 内，依赖 streamlit，
 不在本无栈测试范围；其渲染逻辑已在交互联调中验证，见交接文档。）

运行：
  python test_core.py
退出码：0 = 全绿；1 = 有失败；2 = 依赖缺失（理论上不会发生，本文件零第三方依赖）。
"""
import os
import sys
import json
import tempfile

# query_log 在 import 时读取 QUERY_LOG_PATH，故先指向临时文件再 import
_TMP = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
_TMP.close()
os.environ["QUERY_LOG_PATH"] = _TMP.name

from rag_search_utils import _tokenize, _BM25, _bm25_rank, _rrf_fuse, _stable_ids
import query_log


class Doc:
    """rag_core.Document 的轻量替身：只暴露 page_content + metadata 两属性。"""

    def __init__(self, page_content, source="doc.txt", start_index=0):
        self.page_content = page_content
        self.metadata = {"source": source, "start_index": start_index}


def _build_docs():
    return [
        Doc("事务的ACID特性包括原子性、一致性、隔离性、持久性。", "db.txt", 0),
        Doc("第二范式2NF要求消除部分函数依赖。", "db.txt", 50),
        Doc("死锁是指两个事务互相等待对方释放锁。", "db.txt", 100),
        Doc("苹果是一种水果，富含维生素C。", "fruit.txt", 0),
    ]


def test_tokenize():
    assert _tokenize("ACID") == ["acid"]
    assert _tokenize("死锁") == ["死", "锁"]
    assert _tokenize("2NF 事务隔离级别") == ["2nf", "事", "务", "隔", "离", "级", "别"]
    # 空 / None 不崩
    assert _tokenize("") == []
    assert _tokenize(None) == []


def test_bm25_ranking():
    docs = _build_docs()
    bm25 = _BM25(docs)
    sc = bm25.scores("事务隔离级别 ACID")
    ranked = sorted(range(len(docs)), key=lambda i: sc[i], reverse=True)
    # 含 ACID 的块应排第一（精确术语命中）
    assert "ACID" in docs[ranked[0]].page_content
    assert docs[ranked[0]].metadata["start_index"] == 0
    # 分数单调降序
    vals = [sc[i] for i in ranked]
    assert vals == sorted(vals, reverse=True)
    # 不相关文档（苹果）分数应最低
    assert sc[3] == min(sc)


def test_bm25_rank():
    docs = _build_docs()
    ranked = _bm25_rank("事务隔离级别 ACID", docs, top_n=2)
    assert len(ranked) == 2
    assert "ACID" in ranked[0].page_content
    # top_n=None 返回全部
    assert len(_bm25_rank("死锁", docs)) == 4
    # 空输入
    assert _bm25_rank("x", []) == []


def test_rrf_fuse():
    docs = _build_docs()
    vec_order = list(docs)
    bm25_docs = _bm25_rank("事务隔离级别 ACID", docs, top_n=4)
    fused = _rrf_fuse([vec_order, bm25_docs], top_n=2)
    assert len(fused) == 2
    # 融合分写回 metadata['hybrid_score']
    assert "hybrid_score" in fused[0].metadata
    assert isinstance(fused[0].metadata["hybrid_score"], float)
    # 去重：(source, start_index) 唯一
    keys = [(d.metadata["source"], d.metadata["start_index"]) for d in fused]
    assert len(keys) == len(set(keys))
    # 精确术语块应被融合进 top
    assert any("ACID" in d.page_content for d in fused)
    # top_n 截断生效
    assert len(_rrf_fuse([vec_order, bm25_docs], top_n=1)) == 1


def test_stable_ids_dedup():
    p = "/abs/path/data/db.txt"
    v1 = [Doc("a", p, 0), Doc("b", p, 500)]
    v2 = [Doc("a", p, 0), Doc("b", p, 500)]  # 重传同文件
    ids1, ids2 = _stable_ids(v1), _stable_ids(v2)
    assert ids1 == ids2, "同文件两次切分 id 必须一致（增量 upsert 去重前提）"
    assert len(set(ids1)) == 2, "不同块 id 不应冲突"
    # 不同 start_index → 不同 id
    assert _stable_ids([Doc("a", p, 0)])[0] != _stable_ids([Doc("a", p, 999)])[0]


def test_is_grounded():
    assert query_log._is_grounded("ACID 包括原子性等[1]。") is True
    assert query_log._is_grounded("") is False
    assert query_log._is_grounded(None) is False
    assert query_log._is_grounded("根据提供的资料无法回答。") is False


def test_log_query_format():
    query_log.log_query(
        "什么是 ACID？", "精确",
        [{"rank": 1, "source": "db.txt", "score": 0.92},
         {"rank": 2, "source": "db.txt", "score": 0.71}],
        "ACID 包括原子性等[1]。",
    )
    query_log.log_query("量子计算的影响？", "快速", [], "根据提供的资料无法回答。")
    with open(_TMP.name, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) >= 2
    rec1 = json.loads(lines[-2])
    rec2 = json.loads(lines[-1])
    for rec in (rec1, rec2):
        assert set(rec) == {"ts", "question", "mode", "top_sources",
                             "has_citation", "grounded", "answer_len"}
    # 正常可答：接地 + 有引用 + 长度 > 0 + 来源排名正确
    assert rec1["grounded"] is True
    assert rec1["has_citation"] is True
    assert rec1["answer_len"] > 0
    assert rec1["top_sources"][0]["rank"] == 1
    # 资料不足：未接地 + 无引用
    assert rec2["grounded"] is False
    assert rec2["has_citation"] is False


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n结果：{passed} 通过 / {failed} 失败 / 共 {passed + failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    print("=== CourseRAG 核心函数冒烟测试 (test_core.py) ===")
    sys.exit(_run_all())
