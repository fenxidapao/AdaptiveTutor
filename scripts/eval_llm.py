"""LLM 输出质量评估（PLAN 18.5 D1，行业 Evals 落地）。

行业依据：Socratic dialogue 不能单测，但能按 rubric 打分（datasoft 2026）。
本脚本两层：
- check 模式（免费，CI 常跑）：用 evals/golden/outputs.jsonl 自检 rubric 检查器本身——
  给"已知好/坏的 LLM 输出样本"，断言检查器判定与期望一致（防检查器漂移）。
- live 模式（按需，真实 API）：跑 6 个典型场景（讲解/诊断/追问/盲点/推荐/规划），
  rubric 检查 → 输出打分报告（prompt 或模型变更后跑，回归防漂移）。

用法：
  /d/anacoda3/python.exe scripts/eval_llm.py --mode check
  /d/anacoda3/python.exe scripts/eval_llm.py --mode live [--out reports/eval_llm.md]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==================== rubric 检查器（纯规则，无 LLM 依赖） ====================

# 讲解"泄题"特征：练习答案常含的代码模式/关键词
ANSWER_CODE_MARKERS = [
    "print(", "def ", "return ", "=", "for ", "while ", "lambda ",
    "[" , "]", ".append", ".strip", ".split", "if ", "else:",
]
LEAK_KEYWORDS = ["答案是", "正确代码", "参考实现", "答案如下", "标准答案"]


def check_explain_no_answer(text: str) -> tuple[bool, str]:
    """讲解不得直接给出练习答案（代码片段/答案关键词）。"""
    if not text or len(text) < 20:
        return False, "讲解为空或过短（<20 字）"
    for kw in LEAK_KEYWORDS:
        if kw in text:
            return False, f"疑似泄题关键词: {kw}"
    # 代码特征过多（>4 处）视为在贴代码而非讲解
    hits = sum(1 for m in ANSWER_CODE_MARKERS if m in text)
    if hits >= 6:
        return False, f"代码特征过多（{hits} 处），疑似直接给答案代码"
    return True, "OK"


def check_json_parseable(text: str) -> tuple[bool, str]:
    """能从文本提取出合法 JSON 对象。"""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return False, "无 JSON 花括号"
    try:
        json.loads(text[s:e + 1])
        return True, "JSON 可解析"
    except json.JSONDecodeError as err:
        return False, f"JSON 解析失败: {err}"


def check_reason_length(text: str, lo: int = 15, hi: int = 600) -> tuple[bool, str]:
    if not text:
        return False, "为空"
    if len(text) < lo:
        return False, f"过短（{len(text)}<{lo} 字）"
    if len(text) > hi:
        return False, f"过长（{len(text)}>{hi} 字）"
    return True, f"长度 OK（{len(text)} 字）"


def check_report_consistency(report: dict, db_module, db_path: str | None = None) -> tuple[bool, str]:
    """报告数字必须与 assessments 表一致（前后测正确率）。"""
    pre, post = report.get("pre"), report.get("post")
    if pre is None and post is None:
        return True, "无前后测数据（合法）"
    issues = []
    for kind, val in (("pretest", pre), ("posttest", post)):
        if val is None:
            continue
        rows = [a for a in db_module.list_assessments(report.get("user_id", ""),
                                                      db_path=db_path)
                if a["kind"] == kind]
        if not rows:
            issues.append(f"{kind} 表无记录但报告返回 {val}")
            continue
        last = rows[-1]
        if abs(last["score"] - val) > 1e-6:
            issues.append(f"{kind} 报告 {val} != 表 {last['score']}")
    return (True, "与库一致") if not issues else (False, "; ".join(issues))


RUBRICS = {
    "explain_no_answer": check_explain_no_answer,
    "json_parseable": check_json_parseable,
    "reason_length": check_reason_length,
}


# ==================== check 模式：rubric 自检（golden 驱动） ====================

def run_check(golden_path: Path) -> tuple[int, list[str]]:
    """跑 golden 样本，断言检查器判定 == 期望。返回 (通过数, 失败列表)。"""
    fails: list[str] = []
    passed = 0
    with golden_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            fn = RUBRICS[case["rubric"]]
            ok, msg = fn(case["input"])
            expected = case["expect_pass"]
            if ok == expected:
                passed += 1
            else:
                fails.append(f"case#{i} [{case['rubric']}] 期望 {expected} 实得 {ok}: {msg}")
    return passed, fails


# ==================== live 模式：真实 LLM 评估（按需，花钱） ====================

def run_live() -> list[dict]:
    """真实 API 跑 6 场景 + rubric 检查。返回逐项结果。"""
    import db as dbm
    import learning_pack
    from agents import diagnostic, feynman, planner, recommender

    results = []
    graph = learning_pack.load_graph("python")
    kp = graph["_by_id"]["python.list.slice"]

    def add(name, fn, rubric):
        try:
            out = fn()
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = json.dumps(out, ensure_ascii=False)[:2000]
            else:
                text = str(out)
            ok, msg = RUBRICS[rubric](text)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"异常: {type(e).__name__}: {e}"
        results.append({"场景": name, "rubric": rubric, "pass": ok, "msg": msg})

    # 场景1：费曼追问（1-2 句尖锐问题）
    add("费曼追问", lambda: feynman.generate_followup(
        kp, [{"role": "user", "content": "列表切片就是用冒号取一段元素。"}]), "reason_length")

    # 场景2：盲点总结（返回已解析 list，检查非空）
    add("盲点总结", lambda: str(feynman.summarize_gaps(
        kp, [{"role": "user", "content": "切片就是 a[1:3] 取第 1 到第 3 个元素。"}])), "reason_length")

    # 场景3：诊断画像（临时用户，全错记录 → JSON）
    uid = "u_eval"
    dbm.get_user(uid)
    for ex in learning_pack.load_exercises("python")[:6]:
        dbm.log_exercise(uid, ex["ex_id"], ex.get("kp_id", ""), 0, "wrong", "eval", db_path=None)
    add("诊断画像 JSON", lambda: json.dumps(diagnostic.diagnose(uid), ensure_ascii=False),
        "json_parseable")

    # 场景4：推荐理由（JSON）
    add("推荐理由 JSON", lambda: json.dumps(recommender.recommend(uid, "python", top_n=3),
                                            ensure_ascii=False), "json_parseable")

    # 场景5：规划解释（文本，2-3 句）
    add("规划解释", lambda: planner.plan_path(uid, "python")["rationale"], "reason_length")

    # 场景6：讲解不含答案（RAG 不可用时 notes 兜底）
    add("讲解不泄题", lambda: feynman.explain_kp(uid, "python.list.slice", "python"),
        "explain_no_answer")

    # 场景7：报告与库一致
    try:
        from agents import assessor
        r = assessor.report(uid, None)
        ok, msg = check_report_consistency(r, dbm)
        results.append({"场景": "报告与库一致", "rubric": "report_consistency",
                        "pass": ok, "msg": msg})
    except Exception as e:  # noqa: BLE001
        results.append({"场景": "报告与库一致", "rubric": "report_consistency",
                        "pass": False, "msg": f"异常: {type(e).__name__}"})
    return results


def render_live(results: list[dict]) -> str:
    lines = ["# LLM 输出质量评估（live）", "",
             f"- 生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
             "", "| 场景 | rubric | 通过 | 说明 |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['场景']} | {r['rubric']} | {'✅' if r['pass'] else '❌'} | {r['msg']} |")
    total = sum(1 for r in results if r["pass"])
    lines += ["", f"**通过 {total}/{len(results)}**"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="LLM 输出质量评估（PLAN 18.5 D1）")
    ap.add_argument("--mode", choices=["check", "live"], default="check")
    ap.add_argument("--out", default="reports/eval_llm.md")
    args = ap.parse_args()

    if args.mode == "check":
        golden = Path(__file__).resolve().parent.parent / "evals" / "golden" / "outputs.jsonl"
        passed, fails = run_check(golden)
        print(f"rubric 自检：{passed} 通过，{len(fails)} 失败")
        for f_ in fails:
            print("  FAIL", f_)
        sys.exit(1 if fails else 0)
    else:
        results = run_live()
        report = render_live(results)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n[报告已写入 {out}]")
        sys.exit(1 if any(not r["pass"] for r in results) else 0)


if __name__ == "__main__":
    main()
