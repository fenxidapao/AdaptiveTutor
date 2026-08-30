"""
eval_heldout.py — held-out 评估运行器（工程化 P0：可辩护评估 + 门控）

把"同源自评"升级为可防御证据：在 eval_data_heldout 的三组上分别验证。
  - A 组（语料内，30）：复用 eval_local 的四指标，报告绝对 faithfulness / context_recall，
    门控 faithfulness ≥ 0.85、context_recall ≥ 0.80。
  - B 组（语料外/应拒答，20）：验证硬拒答（检索拒 + 接地自检拒），拒答率 ≥ 90% 为门控通过。
  - C 组（对抗/模糊/注入，10）：仅记录拒答/放通行为，不强制判对错（混合期望）。

运行依赖：本机 Ollama（生成 llama3.1:8b + 裁判 qwen2.5:7b + 嵌入 nomic-embed-text）。
  该脚本在云端 CI（无 Ollama）下无法跑，仅本地 / 自托管 runner 执行（见 .github/workflows）。

用法：
  python eval_heldout.py            # 跑 A+B+C 并出汇总 + 门控判定
  python eval_heldout.py --a-only   # 只跑 A 组绝对指标
  python eval_heldout.py --bc-only  # 只跑 B/C 拒答验证
"""
import os
import sys
import json
import csv

# 让 eval_local 把 A 组结果写到独立 CSV，不与 100 题主评估混淆。
os.environ.setdefault("EVAL_CSV_NAME", "eval_results_heldout_A.csv")
os.environ.setdefault("SKIP_REBUILD", "1")  # 复用现有索引（已存在的 chroma_db）

from rag_core import answer_question
from eval_data_heldout import HELDOUT_A, HELDOUT_B, HELDOUT_C
from eval_local import run_evaluation, summarize_and_save, FIELDS


# ---------- A 组：绝对指标 + 门控 ----------
FAITH_GATE = 0.85
RECALL_GATE = 0.80


def run_group_a():
    print("\n===== A 组（语料内，绝对指标）=====")
    res = run_evaluation(HELDOUT_A)
    avg = summarize_and_save(res)
    passed = avg["faithfulness"] >= FAITH_GATE and avg["context_recall"] >= RECALL_GATE
    print(f"\n[A 门控] faithfulness≥{FAITH_GATE} & context_recall≥{RECALL_GATE} "
          f"-> {'PASS ✅' if passed else 'FAIL ❌'}")
    return avg, passed


# ---------- B / C 组：拒答验证 ----------
BC_FIELDS = ["group", "question", "expect_reject", "final_reject",
             "reject_by_retrieval", "reject_by_grounding", "top1_vector_score",
             "top1_rerank_score", "grounded_score", "answer_head"]


def _safe(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) else v


def run_group_bc():
    print("\n===== B/C 组（语料外 / 对抗，拒答验证）=====")
    rows = []
    for grp, items in (("B", HELDOUT_B), ("C", HELDOUT_C)):
        for it in items:
            q = it["question"]
            # 评审 P1（2026-08-30）起走统一问答入口：与 app.py / main.py 共用同一套
            # 「硬拒答 + 接地自检」闸门，不再各自复刻一份拒答逻辑（避免出口间漂移）；
            # use_cache=False 保证评估不吃语义缓存、逐题真实跑闸门。
            res = answer_question(q, mode="accurate", use_cache=False)
            docs = res["docs"] or []
            top_v = docs[0].metadata.get("vector_score") if docs else None
            top_r = docs[0].metadata.get("rerank_score") if docs else None
            answer = res["answer"]

            final_reject = res["rejected"] or res["not_grounded"]
            rows.append({
                "group": grp,
                "question": q,
                "expect_reject": it.get("expect_reject"),
                "final_reject": final_reject,
                "reject_by_retrieval": res["rejected"],
                "reject_by_grounding": res["not_grounded"],
                "top1_vector_score": _safe(top_v),
                "top1_rerank_score": _safe(top_r),
                "grounded_score": _safe(res["grounded_score"]),
                "answer_head": (answer or "").replace("\n", " ")[:80],
            })
            mark = "拒答" if final_reject else "放通"
            print(f"  [{grp}] {mark}  q={q[:40]}  v={_safe(top_v)} r={_safe(top_r)}")

    # 落盘明细
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "eval_results_heldout_BC.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BC_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # B 组拒答率门控
    b_rows = [r for r in rows if r["group"] == "B"]
    n_b = len(b_rows)
    n_reject = sum(1 for r in b_rows if r["final_reject"])
    rate = (n_reject / n_b) if n_b else 0.0
    REJECT_GATE = 0.90
    passed = rate >= REJECT_GATE
    print(f"\n[B 拒答率] {n_reject}/{n_b} = {rate:.2f}  (门控 ≥{REJECT_GATE}) "
          f"-> {'PASS ✅' if passed else 'FAIL ❌'}")
    print(f"[明细] {csv_path}")
    return rate, passed, rows


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    # None = 该组未跑（此前 bug：默认 True 会让 --bc-only 误报 A 组 PASS）
    a_pass = b_pass = None
    if only in ("", "--a-only"):
        _, a_pass = run_group_a()
    if only in ("", "--bc-only"):
        _, b_pass, _ = run_group_bc()

    def _fmt(p):
        if p is None:
            return "未测 ⏭️"
        return "PASS ✅" if p else "FAIL ❌"

    print("\n===== 总门控 =====")
    print(f"  A 绝对指标: {_fmt(a_pass)}")
    print(f"  B 拒答率:   {_fmt(b_pass)}")
    if a_pass is None or b_pass is None:
        print("  >>> 存在未测组：请分别跑 --a-only / --bc-only，或全量跑（不带参数）后看总门控")
        sys.exit(2)
    overall = a_pass and b_pass
    print(f"  >>> {'全部通过，可称「完整落地评估达标」' if overall else '存在未达标项，继续调参/补语料'}")
    sys.exit(0 if overall else 2)


if __name__ == "__main__":
    main()
