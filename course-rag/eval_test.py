"""
eval_test.py — 冻结测试集回归（#18 评估纪律，2026-08-25）

与 eval_heldout.py 的区别：
  - eval_heldout 跑的是「校准 / 开发」组（A/B/C），用于开发期调参与门控；
  - 本脚本跑的是从未参与调参的冻结测试集 TEST_SET，用于对外正式宣布前的最终回归。

纪律：TEST_SET 失败只能改系统（检索 / 提示 / 阈值），不能改测试集、
也不能把测试集当调参对象继续刷指标。

用法：
  python eval_test.py            # 跑冻结测试集（SKIP_REBUILD=1 复用现有索引）
"""
import os

os.environ["EVAL_CSV_NAME"] = "eval_results_test.csv"
os.environ.setdefault("SKIP_REBUILD", "1")

from eval_data_test import TEST_SET
from eval_local import run_evaluation, summarize_and_save


def main():
    print("===== 冻结测试集回归（#18 评估纪律：仅供报告，禁止调参）=====")
    res = run_evaluation(TEST_SET)
    avg = summarize_and_save(res)
    print("\n[纪律提醒] 本测试集从未参与调参；若未达标，请改系统（检索/提示/阈值），"
          "不要改测试集、不要用本集继续调参。")
    return avg


if __name__ == "__main__":
    main()
