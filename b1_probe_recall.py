"""探针 2：B1 的瓶颈到底是「排序」还是「召回」？

探针 1（b1_probe_pool.py）读数：RRF 池里有 4~7 张图，但 **6 道题里 5 道的目标图根本
不在池里**（target_rank=None）。若属实，B1「让 VLM 重排候选图」在这 5 道题上无解——
VLM 没法给一张没送到它面前的图投票。

本探针分离两个变量：
  A. 池宽：candidate_k ∈ {16, 30, 50, 100}，看目标图是否进池；
  B. 通道：专用「图像-only」向量通道（where={"type":"image"}），看目标图在纯图像召回里的名次。

用法（cwd = CourseRAG 根）：
    python b1_probe_recall.py
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

CASES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？", "p267_i00.png"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？", "p045_i00.png"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？", "p267_i00.png"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？", "p045_i00.png"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？", "p167_i00.png"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？", "p082_i00.png"),
]

SWEEP = [16, 30, 50, 100, 200]


def rank_of(docs, name):
    for i, d in enumerate(docs, 1):
        if name in (d.metadata.get("image_path") or ""):
            return i
    return None


def n_img(docs):
    return sum(1 for d in docs if d.metadata.get("type") == "image")


print("=" * 100)
print("A. 池宽扫描（hybrid RRF，带教材过滤）——看目标图能否进池")
print("=" * 100)
for q, target in CASES:
    src = rag_core._resolve_source_filter(q, None)
    line = [f"{target:14s}"]
    for ck in SWEEP:
        pool = rag_core.hybrid_retrieve(q, source_filter=src, candidate_k=ck)
        line.append(f"k={ck}: img={n_img(pool):2d} rank={str(rank_of(pool, target)):>4s}")
    print("  " + " | ".join(line))

print()
print("=" * 100)
print("B. 专用图像通道（向量召回 where={'type':'image'}，全库，k=50）")
print("=" * 100)
for q, target in CASES:
    docs = rag_core.retrieve_vector(q, k=50, where={"type": "image"})
    r = rank_of(docs, target)
    print(f"  {target:14s} 图像通道共 {len(docs)} 张，目标图 rank={r}")

print()
print("=" * 100)
print("C. 图像通道 + 教材过滤（where 同时限定 source 与 type）")
print("=" * 100)
for q, target in CASES:
    src = rag_core._resolve_source_filter(q, None)
    if not src:
        print(f"  {target:14s} 无路由源，跳过")
        continue
    where = {"$and": [{"type": "image"}, {"source": {"$in": src}}]}
    try:
        docs = rag_core.retrieve_vector(q, k=50, where=where)
    except Exception as e:  # noqa: BLE001
        print(f"  {target:14s} where 组合过滤失败：{type(e).__name__}: {e}")
        continue
    r = rank_of(docs, target)
    print(f"  {target:14s} 图像通道共 {len(docs)} 张，目标图 rank={r}")
