"""
semantic_cache.py — P2 语义缓存（2026-08-25）

为什么：精确模式 reranker 延迟高（口径见 EVALUATION.md）+ LLM 生成慢；课程问答"热问题"重复率高时，
按语义相似（embedding 余弦 ≥ 阈值）命中重复 / 改写问句，直接返回历史答案，跳过检索/重排/生成。

设计：
  - 内存 LRU（单进程；多副本需换 Redis，见点评文档路线 β/ε）
  - key = 问题 embedding（nomic-embed-text，每次 ~100ms，远小于 reranker/生成）
  - 相似度阈值 config.SEMANTIC_CACHE_SIM（默认 0.93）：同义改写可命中，意图不同不误命中
  - 索引版本绑定：重建索引后调 invalidate() 使旧缓存全部作废（旧语料上的答案不可复用）
  - 只缓存"正常答案"；硬拒答 / 未接地不缓存（避免把错误拒答当成热答案复用）
  - 线程安全：FastAPI 多线程下读写加锁

用法：
  from semantic_cache import lookup, store, invalidate, stats
  hit = lookup(question, mode)      # -> None 或 {"answer","citations","ts","sim"}
  store(question, mode, answer, citations, reject=False)
  invalidate()                       # 索引重建后调用
"""
import math
import threading
from collections import OrderedDict
from datetime import datetime

import ollama

import config

_lock = threading.Lock()
_cache = OrderedDict()  # (question, mode) -> entry，按插入序做 LRU 淘汰
_version = 0            # 索引版本：重建后 +1，旧缓存全部作废
_client = ollama.Client(host=config.OLLAMA_BASE_URL)


def _embed(question):
    """查询嵌入；Ollama 不可用返回 None，缓存自动降级为不缓存（不影响主链路）。"""
    try:
        resp = _client.embed(model=config.EMBEDDING_MODEL, input=question)
        return resp["embeddings"][0]
    except Exception:
        return None


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def invalidate():
    """索引重建后调用：旧缓存基于旧语料，全部作废。"""
    global _version
    with _lock:
        _version += 1
        _cache.clear()


def lookup(question, mode):
    """语义缓存查找：命中返回 {answer, citations, ts, sim}，未命中返回 None。"""
    if not config.SEMANTIC_CACHE_ENABLED:
        return None
    qv = _embed(question)
    if qv is None:
        return None
    with _lock:
        best = None
        best_sim = 0.0
        for (q, m), e in _cache.items():  # noqa: B007
            if m != mode:
                continue
            s = _cosine(qv, e["vec"])
            if s >= config.SEMANTIC_CACHE_SIM and s > best_sim:
                best = e
                best_sim = s
    if best is None:
        return None
    return {"answer": best["answer"], "citations": best["citations"],
            "ts": best["ts"], "sim": best_sim}


def store(question, mode, answer, citations, reject=False):
    """写入缓存。拒答 / 未接地 / 空答案不缓存；超限按 LRU 淘汰最旧。"""
    if not config.SEMANTIC_CACHE_ENABLED or reject:
        return
    if not answer or not answer.strip():
        return
    qv = _embed(question)
    if qv is None:
        return
    entry = {
        "vec": qv,
        "answer": answer,
        "citations": citations,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
    }
    with _lock:
        key = (question, mode)
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > config.SEMANTIC_CACHE_MAX:
            _cache.popitem(last=False)


def stats():
    with _lock:
        return {"size": len(_cache), "max": config.SEMANTIC_CACHE_MAX, "version": _version}
