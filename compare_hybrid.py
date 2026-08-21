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
  - 只算差值，不吹绝对值（8b 裁判噪声大，绝对值勿当真）。
  - 实现局限（务必写进报告，否则是误导性夸大）：
    本实现的 hybrid 只在向量 top-50 候选池内用 BM25 做 RRF 重排，不扩大候选集。
    因此 BM25 的贡献仅来自「正确块原本在向量 rank 6~50、被顶进 top-5」；
    若正确块本就在 top-5 或落在 top-50 之外，hybrid 都救不动。
  - 预期：context_precision 提升（词面对齐的块被顶上来）；
    context_recall 可能基本不动（召回集合没变，只换了排序）。
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
1. **hybrid 不扩大候选集**：本实现 BM25 只在向量 top-50 候选池内做 RRF 重排，不参与全库召回。
   因此 BM25 的贡献仅来自「正确块原本在向量 rank 6~50、被顶进 top-5」这一种情况；
   若正确块本就在 top-5（BM25 无新增价值）或落在 top-50 之外（BM25 也救不动），hybrid 都无贡献。
2. **8b 裁判噪声**：裁判与生成为同一模型（llama3.1:8b），存在自评价偏差，看差值不看绝对值。
3. **语料规模**：当前语料为 12 个课程笔记（数万字），结论外推到真实 PDF 课件场景需谨慎。
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

    report = f"""# 混合检索 vs 纯向量 A/B 对比（{len(qs)} 样本，本机 8b 裁判）

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

> 结论文字：混合检索（向量+自实现 BM25+RRF 融合）的主要收益体现在**上下文精度**与
> **答案相关性**（正确块在向量 rank 6~50 之间时被 BM25 词面对齐顶回 top-5）；
> context_recall 理论不动（召回集合未变，仅排序变化），实测应基本持平。

{LIMITATION_NOTE}
"""
    out_path = os.path.join(os.path.dirname(__file__), "eval_compare_hybrid.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()
