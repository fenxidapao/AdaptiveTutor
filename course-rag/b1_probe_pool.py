"""探针：B1 候选池里到底有几张图？

动机（2026-10-14）：B1 原设计「只重排图像块，文本块位置不动」有一个致命前提——
候选里得有 ≥2 张图，否则 `len(image_docs) < 2` 直接跳过，视觉重排**静默失效**。
而 reranker 截断到 RERANKER_TOP_N=8 之后，图像块往往只剩 0~1 个。

本探针回答三个数：
  1. 各阶段（RRF 池 / reranker 后 top-8）里图像块的数量；
  2. 6 道基准题的目标图在各阶段的名次；
  3. 若把 reranker 不截断（保留 RERANK_CANDIDATE_K=16），目标图能否进池。

用法（cwd = CourseRAG 根）：
    python b1_probe_pool.py
"""
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _load_vlm_key():
    if os.environ.get("VISUAL_RERANK_API_KEY"):
        return
    cred = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
    if cred.is_file():
        m = re.search(r"KUAIPAO_API_KEY:\s*(\S+)", cred.read_text(encoding="utf-8"))
        if m:
            os.environ["VISUAL_RERANK_API_KEY"] = m.group(1)


_load_vlm_key()

import config  # noqa: E402
import rag_core  # noqa: E402
from reranker import rerank_documents  # noqa: E402

CASES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？", "p267_i00.png"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？", "p045_i00.png"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？", "p267_i00.png"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？", "p045_i00.png"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？", "p167_i00.png"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？", "p082_i00.png"),
]


def stats(docs):
    imgs = [d for d in docs if d.metadata.get("type") == "image"]
    return len(docs), len(imgs)


def rank_of(docs, name):
    for i, d in enumerate(docs, 1):
        if name in (d.metadata.get("image_path") or ""):
            return i
    return None


print(f"RETRIEVE_K={config.RETRIEVE_K} RERANK_CANDIDATE_K={config.RERANK_CANDIDATE_K} "
      f"RERANKER_TOP_N={config.RERANKER_TOP_N}")

for q, target in CASES:
    print("=" * 96)
    print(f"Q: {q}\n   target={target}")
    sources = rag_core._resolve_source_filter(q, None)
    print(f"   routed sources = {[pathlib.Path(s).name for s in sources] if sources else None}")

    # 阶段 1：RRF 融合池（未过 reranker）
    pool = rag_core.hybrid_retrieve(q, source_filter=sources)
    n, ni = stats(pool)
    print(f"   [1] RRF pool        : {n:3d} docs, {ni:2d} images, target_rank={rank_of(pool, target)}")

    # 阶段 2：现网路径（reranker 截到 top-8）
    top8 = rerank_documents(q, pool, top_n=config.RERANKER_TOP_N)
    n, ni = stats(top8)
    print(f"   [2] reranker top-8  : {n:3d} docs, {ni:2d} images, target_rank={rank_of(top8, target)}")

    # 阶段 3：不截断，看整池
    full = rerank_documents(q, pool, top_n=None)
    n, ni = stats(full)
    print(f"   [3] reranker no-cut : {n:3d} docs, {ni:2d} images, target_rank={rank_of(full, target)}")

    # 阶段 4：无教材过滤的全库 RRF 池
    pool_all = rag_core.hybrid_retrieve(q, source_filter=None)
    n, ni = stats(pool_all)
    print(f"   [4] pool(no filter): {n:3d} docs, {ni:2d} images, target_rank={rank_of(pool_all, target)}")
