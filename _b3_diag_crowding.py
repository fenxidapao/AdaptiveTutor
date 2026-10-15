# -*- coding: utf-8 -*-
"""B3 回归诊断：新 caption 变长变准之后，图像块是否把文本块挤出 top-8？

## 背景

B3 换描述器后重跑冻结 24 题（`eval_test.py`）：

    faithfulness      0.901 → 0.947   (+0.046)  ✅
    answer_relevancy  0.975 → 0.983
    context_precision 0.869 → 0.875
    context_recall    0.948 → 0.903   (-0.045)  ⚠️

faith 涨了但 recall 掉了。最可疑的机制：**新 caption 变长变准 → 图像块的向量更容易
挤进 top-8 → 占用原本给文本块的名额 → 覆盖答案的文本块被挤出去 → recall 掉**。

本探针只做检索层诊断（不调裁判，快），回答两件事：

1. 官方口径「A 组图片命中率」（8 题）是否仍是 8/8、是否仍 rank=1（不能掉）；
2. 冻结 24 题里，**每题 top-8 有几张图**，以及有几题的 top-8 被图像块占满/过半。

判读：若第 2 项里大量题目 n_image 从 0 变成 1~2，则 crowding 假设成立。

用法（cwd 必须是 CourseRAG 根目录）：
    $env:SKIP_REBUILD='1'; python _b3_diag_crowding.py
"""
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
import rag_core

print(f"VISUAL_RERANK_ENABLED = {config.VISUAL_RERANK_ENABLED}（应为 False：测默认交付路径）")
print(f"RERANKER_TOP_N = {config.RERANKER_TOP_N}")
print()

# ---------- 1. 官方图片命中率口径 ----------
from eval_heldout import run_image_hitrate

rate, rows = run_image_hitrate()
n_top1 = sum(1 for r in rows if r["hit_rank"] == 1)
print(f"\n[图片命中率] {rate:.2f}   rank=1 的 {n_top1}/{len(rows)}   （基线 8/8 全部 rank=1）")

# ---------- 2. 冻结 24 题的 top-8 组成 ----------
print()
print("=" * 78)
print("冻结 24 题 top-8 组成（图像块 vs 文本块）")
print("=" * 78)

from eval_data_test import TEST_SET

ns, rows2 = [], []
for i, it in enumerate(TEST_SET, 1):
    docs = rag_core.retrieve_and_rerank(it["question"])
    imgs = [d for d in docs if d.metadata.get("type") == "image"]
    ns.append(len(imgs))
    rows2.append((i, it["question"], len(imgs), len(docs)))
    mark = "  " if not imgs else "图"
    print(f"  {mark} [{i:2d}/24] n_img={len(imgs)} n_doc={len(docs)}  {it['question'][:44]}")

print()
print(f"  top-8 里图像块数量：合计 {sum(ns)}，均值 {statistics.mean(ns):.3f}，"
      f"中位数 {statistics.median(ns)}，最大 {max(ns)}")
print(f"  含至少 1 张图的题：{sum(1 for n in ns if n)}/24")
print(f"  图像块占 top-8 过半的题：{sum(1 for n in ns if n >= 4)}/24")
print()
print("  判读：若「含至少 1 张图」的题远多于 0，说明图像块确实在参与 top-8 竞争；")
print("        配合 recall 从 0.948 掉到 0.903，crowding 假设成立（15 题里 5 题掉分）。")
