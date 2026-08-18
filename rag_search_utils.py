"""
rag_search_utils.py — 零依赖的纯检索算法（P2-1 可单测核心）

把混合检索里的「纯算法」从 rag_core.py 抽出来，单独成模块：
  - 只依赖标准库（re / math / hashlib），不 import chromadb / ollama / sentence_transformers；
  - 因此可以在没有 LLM 栈的环境里（CI、沙箱、本机 venv 外）直接 import 并单测；
  - rag_core.py 通过 `from rag_search_utils import (...)` 重新导出，对外接口不变，
    内部调用（hybrid_retrieve / add_documents）照旧工作。

包含的纯函数：
  _tokenize(text)            中英文混合分词（拉丁串整词 + CJK 逐字）
  _BM25                       Okapi BM25 打分（自实现，无外部依赖）
  _bm25_rank(question,docs)   对同一批候选做 BM25 词面打分并降序返回
  _rrf_fuse(ranked_lists,N)   倒数排名融合（RRF），写回 metadata['hybrid_score']
  _stable_ids(chunks)         稳定 id = md5(source::start_index)，用于增量 upsert 去重

文档对象约定：只需有 .page_content（str）与 .metadata（dict）两个属性，
因此单测里用轻量 duck-typed 对象即可，无需引入 langchain / rag_core.Document。
"""
import re
import math
import hashlib


def _tokenize(text: str):
    """中英文混合分词，零新依赖：
    - 连续拉丁/数字串（ACID、2NF、transaction）作为一个 token（转小写）
    - 每个 CJK 汉字作为独立 unigram（无 jieba，但足以覆盖课程精确术语的词面对齐）
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens += re.findall(r"[一-鿿]", text)
    return tokens


class _BM25:
    """Okapi BM25（自实现，无外部依赖）。在给定候选文档集合上打分。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(d.page_content) for d in docs]
        self.n = len(self.doc_tokens)
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        self.df = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def scores(self, query: str):
        q_tokens = _tokenize(query)
        if not q_tokens or self.n == 0:
            return [0.0] * self.n
        out = []
        for i, toks in enumerate(self.doc_tokens):
            dl = self.doc_lens[i]
            freq = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for qt in set(q_tokens):
                f = freq.get(qt, 0)
                if f == 0 or qt not in self.df:
                    continue
                idf = math.log((self.n - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
                score += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out.append(score)
        return out


def _bm25_rank(question, docs, top_n=None):
    """对 docs（同一批向量候选）做 BM25 词面打分，返回按分数降序的 Document 列表。"""
    if not docs:
        return []
    bm25 = _BM25(docs)
    scores = bm25.scores(question)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    if top_n:
        ranked = ranked[:top_n]
    return [docs[i] for i in ranked]


def _rrf_fuse(ranked_lists, top_n, rrf_k=60):
    """倒数排名融合（Reciprocal Rank Fusion）。
    ranked_lists: 多个已按相关性降序排列的 Document 列表（排名从 0 起）。
    同一文档用 (source, start_index) 去重，融合分 = Σ 1/(rank + rrf_k)，降序取 top_n。
    融合分写回 doc.metadata['hybrid_score']（供 UI 在 reranker 关闭时展示）。
    """
    fused = {}
    for rl in ranked_lists:
        for rank, doc in enumerate(rl):
            key = (doc.metadata.get("source"), doc.metadata.get("start_index"))
            entry = fused.get(key)
            if entry is None:
                entry = {"doc": doc, "score": 0.0}
                fused[key] = entry
            entry["score"] += 1.0 / (rank + rrf_k)
    items = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    if top_n:
        items = items[:top_n]
    for it in items:
        it["doc"].metadata["hybrid_score"] = round(it["score"], 4)
    return [it["doc"] for it in items]


def _stable_ids(chunks):
    """稳定 id = md5(source::start_index)。同一文件同一块 → 同一 id，upsert 天然去重。"""
    return [
        hashlib.md5(
            f"{d.metadata['source']}::{d.metadata['start_index']}".encode("utf-8")
        ).hexdigest()
        for d in chunks
    ]
