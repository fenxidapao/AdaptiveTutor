"""
_p1a_compare.py — P1a 教材过滤 A 组校准对比（一次性分析工具，2026-09-04）

用法：
  python _p1a_compare.py [旧CSV] [新CSV]
默认：旧 = eval_results_heldout_A_baseline_20260904.csv
      新 = eval_results_heldout_A.csv

输出：
  1. 四指标总体对比（faith / relevancy / precision / recall）
  2. 逐题分差（recall 变化 + faith 变化，按 |Δrecall|+|Δfaith| 降序）
  3. 门控判定（faith≥0.85 & recall≥0.80）
"""
import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

OLD = sys.argv[1] if len(sys.argv) > 1 else "eval_results_heldout_A_baseline_20260904.csv"
NEW = sys.argv[2] if len(sys.argv) > 2 else "eval_results_heldout_A.csv"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for m in METRICS:
            try:
                r[m] = float(r.get(m, 0.0) or 0.0)
            except (TypeError, ValueError):
                r[m] = 0.0
    return {r["question"]: r for r in rows}, rows


old_map, old_rows = load(OLD)
new_map, new_rows = load(NEW)

print(f"旧 = {OLD}  ({len(old_rows)} 题)")
print(f"新 = {NEW}  ({len(new_rows)} 题)")

print("\n===== 总体对比 =====")
print(f"{'指标':<20}{'旧':>8}{'新':>8}{'Δ':>8}")
for m in METRICS:
    o = sum(r[m] for r in old_rows) / len(old_rows)
    n = sum(r[m] for r in new_rows) / len(new_rows)
    mark = "  ←" if abs(n - o) >= 0.02 else ""
    print(f"{m:<20}{o:>8.3f}{n:>8.3f}{n - o:>+8.3f}{mark}")

o_faith = sum(r["faithfulness"] for r in old_rows) / len(old_rows)
o_rec = sum(r["context_recall"] for r in old_rows) / len(old_rows)
n_faith = sum(r["faithfulness"] for r in new_rows) / len(new_rows)
n_rec = sum(r["context_recall"] for r in new_rows) / len(new_rows)
print(f"\n[A 门控] faith≥0.85 & recall≥0.80")
print(f"  旧: faith={o_faith:.3f} recall={o_rec:.3f}")
print(f"  新: faith={n_faith:.3f} recall={n_rec:.3f}")
ok = n_faith >= 0.85 and n_rec >= 0.80
not_down = n_faith >= o_faith - 0.02 and n_rec >= o_rec - 0.02
print(f"  门控: {'PASS ✅' if ok else 'FAIL ❌'}   相对基线: {'不降 ✅' if not_down else '有回归 ⚠️'}")

print("\n===== 逐题分差（|Δrecall|+|Δfaith| 降序，前 15）=====")
diffs = []
for q, nr in new_map.items():
    orow = old_map.get(q)
    if not orow:
        continue
    dr = nr["context_recall"] - orow["context_recall"]
    df = nr["faithfulness"] - orow["faithfulness"]
    diffs.append((abs(dr) + abs(df), dr, df, q, orow, nr))
diffs.sort(reverse=True)
for mag, dr, df, q, orow, nr in diffs[:15]:
    arrow = "↑" if (dr + df) > 0 else ("↓" if (dr + df) < 0 else "→")
    print(f"  {arrow} Δr={dr:+.2f} Δf={df:+.3f}  "
          f"[{orow['context_recall']}→{nr['context_recall']} / {orow['faithfulness']}→{nr['faithfulness']}]  {q[:42]}")

print("\n===== 分差汇总 =====")
up = sum(1 for _, dr, df, *_ in diffs if dr + df > 0.05)
down = sum(1 for _, dr, df, *_ in diffs if dr + df < -0.05)
same = len(diffs) - up - down
print(f"  改善 {up} 题 / 回归 {down} 题 / 基本持平 {same} 题（阈值 |Δ|>0.05）")
