"""
reranker.py — 基于 bge-reranker 的跨编码器重排（cross-encoder）。

为什么需要它？
  向量召回是『双塔』粗排：query 和 doc 各自编码成向量再算余弦相似度，
  只看整体语义接近度，分不出『词序 / 否定 / 细粒度匹配』。
  reranker 是『单塔交互』精排：把 (query, doc) 拼在一起喂给模型，
  能捕捉细粒度相关性，把真正相关的段落顶上来。

这是生产 RAG 的标配两段式（召回 + 重排），QAnything / ThinkRAG 都这么做。

依赖与风险：
  - 需要 sentence-transformers（连带装 torch，体积大）
  - 首次会从 HuggingFace 下载模型权重（bge-reranker-v2-m3 约 2.2GB，需联网一次，之后离线可用）
  - 因此 import 延迟到函数内：RERANKER_ENABLED=false 时完全不触发，保证能先跑通
"""
import config


_reranker = None


def _resolve_device() -> str:
    """解析精排设备：auto=有 CUDA 用 GPU，否则 CPU；也支持强制 cuda/cpu（config.RERANKER_DEVICE）。"""
    if config.RERANKER_DEVICE != "auto":
        return config.RERANKER_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_reranker():
    """懒加载 cross-encoder，进程内只实例化一次。

    2026-09-03 GPU 化改造：设备由 config.RERANKER_DEVICE 决定（auto 时探测 CUDA）。
    前提是 venv 里装了 CUDA 版 torch（CPU 轮子时 auto 会静默回退 CPU，行为与旧版一致）。
    """
    global _reranker
    if _reranker is None:
        # trust_remote_code=True：bge-reranker-v2-m3 含自定义模型代码，必须开
        from sentence_transformers import CrossEncoder

        device = _resolve_device()
        _reranker = CrossEncoder(config.RERANKER_MODEL, trust_remote_code=True, device=device)
        # fp16 默认关：分数会轻微漂移，硬拒答阈值按 fp32 校准，开启前须重验（见 config 注释）
        if device == "cuda" and config.RERANKER_FP16:
            _reranker.model.half()
        print(f"[reranker] device={device}"
              + (" (fp16)" if device == "cuda" and config.RERANKER_FP16 else ""))
    return _reranker


def rerank_documents(query, docs, top_n=None):
    """
    对召回文档按与 query 的相关性打分并降序排序。
    把分数写回每个 doc.metadata['rerank_score']，供 UI 展示与后续评估复用。
    返回按分数降序的 Document 列表。
    """
    if not docs:
        return docs

    reranker = get_reranker()
    # B3 修复（2026-10-16）：图像块优先用 metadata['embed_text']（v1 短描述）打分。
    # 实测：新描述器的逐字转写（中位 264 字、含本机路径等噪声）会让 cross-encoder
    # 给图像块更低的分，被文本块挤出 top-8（v1 短描述库可复现 8/8 rank=1）。
    # 文本块无 embed_text，仍用 page_content。
    pairs = [(query, doc.metadata.get("embed_text") or doc.page_content) for doc in docs]
    scores = reranker.predict(pairs, batch_size=config.RERANKER_BATCH_SIZE)  # 一维数组，值越大越相关

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_n:
        scored = scored[:top_n]

    for doc, s in scored:
        doc.metadata["rerank_score"] = float(s)

    return [doc for doc, _ in scored]
