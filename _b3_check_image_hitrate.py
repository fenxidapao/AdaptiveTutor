# -*- coding: utf-8 -*-
"""B3 回归探针：只跑官方「A 组图片命中率」口径，不跑裁判（省 10 分钟）。

`eval_heldout.py` 的 `run_image_hitrate()` 是仓库自带的 P2 多模态口径：
期望图像块（`expected_image` 子串匹配 `image_path`）是否进检索 top-8，8 道题。

基线（B3 之前，文档记录 + 实测）：**8/8 = 1.00，且全部 rank=1**。

B3 换了全部 215 条 caption 并重建索引，所以必须复跑这个口径确认没掉。

本探针 **不改 config**：`VISUAL_RERANK_ENABLED` 保持默认 False，测的就是
「B3 之后、未开视觉重排」的默认交付路径。

用法（cwd 必须是 CourseRAG 根目录）：
    $env:SKIP_REBUILD='1'; python _b3_check_image_hitrate.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config

print(f"VISUAL_RERANK_ENABLED = {config.VISUAL_RERANK_ENABLED}（默认交付路径应为 False）")
print(f"IMAGE_CAPTIONS_CSV    = {config.IMAGE_CAPTIONS_CSV}")
print()

from eval_heldout import run_image_hitrate

rate, rows = run_image_hitrate()

print()
print("=== 逐题 ===")
for r in rows:
    print(f"  {'HIT ' if r['hit_top8'] else 'MISS'} rank={r['hit_rank']}  "
          f"img_in_top8={r['n_image_in_top8']}  {r['expected_image']}")

n_top1 = sum(1 for r in rows if r["hit_rank"] == 1)
print()
print(f"图片命中率 top-8 = {rate:.2f}   其中 rank=1 的 {n_top1}/{len(rows)}")
print(f"基线对照：B3 之前 8/8 = 1.00，全部 rank=1")
print(f"结论：{'✅ 无回归' if rate >= 1.0 else '❌ 有回归，需查'}")
