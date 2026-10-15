# -*- coding: utf-8 -*-
"""B3 的 A/B 对照：用**旧 caption + 旧索引**复跑冻结 24 题，隔离「换描述器」这一件事。

## 为什么要做

B3 之后冻结 24 题的四个指标里，faith 涨了但 recall 掉了：

    faithfulness      0.901 → 0.947   (+0.046)  ✅
    context_recall    0.948 → 0.903   (-0.045)  ⚠️

但基线 0.901±0.022 是 3 轮的标准差；recall 的 0.948 只有单点读数，**不知道噪声带宽**。
不确定「-0.045 是真回归还是抖动」就不能对外说 B3 干净。

## 怎么隔离

`IMAGE_CAPTIONS_CSV` 与 `PERSIST_DIRECTORY` 都是环境变量可覆盖的，所以可以
**不动主索引**（`chroma_db` 仍是 B3 的），另建一份旧索引做对照：

    PERSIST_DIRECTORY=./chroma_db_v1
    IMAGE_CAPTIONS_CSV=_image_captions_v1_qwen25vl.csv

代码路径完全一致（`VISUAL_RERANK_ENABLED` 默认 False，B1 不参与），
唯一差异就是 caption —— 干净的单变量对照。

用法（cwd 必须是 CourseRAG 根目录）：
    $env:PERSIST_DIRECTORY='./chroma_db_v1'
    $env:IMAGE_CAPTIONS_CSV='_image_captions_v1_qwen25vl.csv'
    python _b3_rebuild.py          # 先建旧索引
    python _b3_ab_baseline.py      # 再跑冻结 24 题
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("SKIP_REBUILD", "1")
# 不覆盖 B3 的结果文件，另存一份
os.environ["EVAL_CSV_NAME"] = "eval_results_test_v1captions.csv"

import config  # noqa: E402

print("=== B3 A/B 对照组（旧 caption + 旧索引）===")
print(f"PERSIST_DIRECTORY  = {config.PERSIST_DIRECTORY}")
print(f"IMAGE_CAPTIONS_CSV = {config.IMAGE_CAPTIONS_CSV}")
print(f"VISUAL_RERANK_ENABLED = {config.VISUAL_RERANK_ENABLED}")
print()

from eval_data_test import TEST_SET  # noqa: E402
from eval_local import run_evaluation, summarize_and_save  # noqa: E402


def main():
    res = run_evaluation(TEST_SET)
    avg = summarize_and_save(res)
    print("\n=== 对照完成 ===")
    print("与 B3 组逐项对比：")
    print("  faithfulness      基线 ??  B3 0.947")
    print("  answer_relevancy  基线 ??  B3 0.983")
    print("  context_precision 基线 ??  B3 0.875")
    print("  context_recall    基线 ??  B3 0.903   <- 看这里：旧 caption 是否也在 0.90 附近")
    return avg


if __name__ == "__main__":
    main()
