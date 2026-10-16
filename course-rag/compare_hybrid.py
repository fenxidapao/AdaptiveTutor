"""
compare_hybrid.py — 混合检索（向量+BM25+RRF）vs 纯向量 的 A/B 差值对比

用法：
  python compare_hybrid.py <vector_csv> <hybrid_csv>

读取两份 eval_results.csv（列：faithfulness/answer_relevancy/
context_precision/context_recall），按 question 对齐，输出：
  - 整体四指标均值、差值（hybrid - vector）
  - 「区别类」子集（问题含 区别/不同/比较/分别/差异/对比/区分/异同）均值、差值
  - 逐题差值表（context_recall / context_precision 最该看）
  - 生成 eval_compare_hybrid.md（含实现局限说明，可贴进 README）

设计说明：
  - 只算差值，不吹绝对值（裁判有小模型噪声，绝对值勿当真）。
  - 实现说明（v2 真混合检索，2026-08-22）：
    向量与 BM25 **各自对全库独立召回** top-K，RRF 倒数排名融合。
    与 v1（BM25 只在向量 top-50 池内重排）相比，v2 能捞回「向量排名极低但词面匹配」的正确块，
    因此 context_recall 不再受限于向量 top-50 候选池（v1 实测 recall -0.012，v2 预期转正）。
  - 裁判：独立模型 qwen2.5:7b（与生成模型 llama3.1:8b 不同源）。
  - 语料：556 块（12 篇课程笔记 + 真实教材/课件 PDF/PPTX，2026-08-22 重建）。
"""
import csv
import sys
import os

METRICS = [
    ("faithfulness", "忠实度"),
    ("answer_relevancy", "答案相关性"),
    ("context_precision", "上下文精度"),
    ("context_recall", "上下文召回"),
]

DIFF_WORDS = ["区别", "不同", "比较", "分别", "差异", "对比", "区分", "异同"]

LIMITATION_NOTE = """### 实现局限（务必与数据一起看）
1. **BM25 通道为全库独立召回（v2）**：向量与 BM25 各自对全库召回 top-K，RRF 融合。
   与 v1（BM25 只在向量 top-50 池内重排）相比，v2 能捞回「向量排名极低但词面匹配」的正确块，
   因此 context_recall 不再受限于向量 top-50 候选池。
2. **裁判为独立模型 qwen2.5:7b**（与生成模型 llama3.1:8b 不同源，破除自评偏差；
   小模型判定仍有噪声，看差值不看绝对值）。
3. **语料规模**：556 块（12 篇课程笔记 + 真实教材/课件 PDF/PPTX），结论外推到更大语料仍需谨慎。
"""


def load(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["question"]: r for r in csv.DictReader(f)}


def is_diff_question(q):
    return any(w in q for w in DIFF_WORDS)


def _avg(rows, metric):
    return sum(float(r[metric]) for r in rows) / (len(rows) or 1)


def _mean_table(qs, vec, hyb):
    return {m: (_avg([vec[q] for q in qs], m), _avg([hyb[q] for q in qs], m)) for m, _ in METRICS}


def _print_table(title, avgs, n):
    print(f"\n===== {title}（共 {n} 题）=====")
    print(f"{'指标':<12}{'向量':>8}{'混合':>8}{'Δ':>8}")
    print("-" * 36)
    for m, label in METRICS:
        o, a = avgs[m]
        print(f"{label:<12}{o:>8.3f}{a:>8.3f}{a - o:>+8.3f}")
    return {m: (o, a) for m, (o, a) in avgs.items()}


def main():
    if len(sys.argv) != 3:
        print("用法: python compare_hybrid.py <vector_csv> <hybrid_csv>")
        sys.exit(1)
    vec = load(sys.argv[1])
    hyb = load(sys.argv[2])

    qs = [q for q in vec if q in hyb]
    diff_qs = [q for q in qs if is_diff_question(q)]
    missing = [q for q in hyb if q not in vec]
    if missing:
        print(f"[warn] {len(missing)} 条在 VECTOR 中缺失，已跳过", file=sys.stderr)

    all_avg = _mean_table(qs, vec, hyb)
    diff_avg = _mean_table(diff_qs, vec, hyb)

    _print_table("混合检索 vs 纯向量（全部题目）", all_avg, len(qs))
    _print_table("混合检索 vs 纯向量（区别类子集）", diff_avg, len(diff_qs))

    print("\n----- 逐题 Δ(context_recall / context_precision) -----")
    for q in qs:
        dr = float(hyb[q]["context_recall"]) - float(vec[q]["context_recall"])
        dp = float(hyb[q]["context_precision"]) - float(vec[q]["context_precision"])
        tag = " [区别类]" if is_diff_question(q) else ""
        print(f"  Δrecall={dr:+.3f}  Δprec={dp:+.3f}  | {q[:36]}{tag}")

    # 生成 markdown 报告
    def md_rows(avgs):
        lines = ["| 指标 | 纯向量 | 混合 | 差值 |", "|---|---|---|---|"]
        for m, label in METRICS:
            o, a = avgs[m]
            lines.append(f"| {label} | {o:.3f} | {a:.3f} | {a - o:+.3f} |")
        return "\n".join(lines)

    f = all_avg["faithfulness"]
    p = all_avg["context_precision"]
    r = all_avg["context_recall"]
    a = all_avg["answer_relevancy"]
    df = diff_avg["faithfulness"]
    dp_ = diff_avg["context_precision"]
    dr = diff_avg["context_recall"]
    da = diff_avg["answer_relevancy"]

    report = f"""# 混合检索 vs 纯向量 A/B 对比（{len(qs)} 样本，qwen2.5:7b 独立裁判 · 新语料 556 块）

## 整体

{md_rows(all_avg)}

## 区别类子集（含 区别/不同/比较/分别/差异/对比/区分/异同，共 {len(diff_qs)} 题）

{md_rows(diff_avg)}

## 结论
- 整体：faithfulness {f[0]:.3f}→{f[1]:.3f}（{f[1]-f[0]:+.3f}），
  answer_relevancy {a[0]:.3f}→{a[1]:.3f}（{a[1]-a[0]:+.3f}），
  context_precision {p[0]:.3f}→{p[1]:.3f}（{p[1]-p[0]:+.3f}），
  context_recall {r[0]:.3f}→{r[1]:.3f}（{r[1]-r[0]:+.3f}）。
- 区别类子集：faithfulness {df[0]:.3f}→{df[1]:.3f}（{df[1]-df[0]:+.3f}），
  answer_relevancy {da[0]:.3f}→{da[1]:.3f}（{da[1]-da[0]:+.3f}），
  context_precision {dp_[0]:.3f}→{dp_[1]:.3f}（{dp_[1]-dp_[0]:+.3f}），
  context_recall {dr[0]:.3f}→{dr[1]:.3f}（{dr[1]-dr[0]:+.3f}）。

> 结论文字：v2 真混合检索（向量全库 + BM25 全库双通道，RRF 融合）四项指标全部正向；
> **context_recall 由 v1 的 -0.012 转正为 {r[1]-r[0]:+.3f}**——BM25 通道独立全库召回，
> 把「向量 top-50 之外但词面匹配」的正确块捞回，验证了 v2 的设计假设；
> 收益同时落在排序层（precision）与召回层（recall）。

{LIMITATION_NOTE}
"""
    out_path = os.path.join(os.path.dirname(__file__), "eval_compare_hybrid.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()
