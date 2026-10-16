"""
eval_data_test.py — #18 评估过拟合防护：冻结测试集（2026-08-25）

背景：HELDOUT_A（30）在 2026-08-24 调参（RERANKER_TOP_N 5→8 等）时用过，
已降级为「校准集」。对外正式宣布 / 最终回归前，需要一组从未被调参碰过的冻结测试集。

构造：从 eval_data.EVAL_SAMPLES（100 题）剔除已进 HELDOUT_A 的题，按固定步长抽样
（确定性、可复现），得 TEST_SET。
**纪律：TEST_SET 只用于报告 / 回归，禁止用它调参；它不过只能改系统，不能改测试集。**
"""
from eval_data import EVAL_SAMPLES
from eval_data_heldout import HELDOUT_A

_A_QS = {a["question"] for a in HELDOUT_A}
_NON_A = [s for s in EVAL_SAMPLES if s["question"] not in _A_QS]

# 固定步长抽样，覆盖各主题段（确定性、可复现，避免与 A 组重叠）
TEST_SET = [_NON_A[i] for i in range(0, len(_NON_A), 3)]

if __name__ == "__main__":
    print(f"测试集：{len(TEST_SET)} 条（100 题剔除 A 组 30 后按步长 3 抽样，冻结）")
