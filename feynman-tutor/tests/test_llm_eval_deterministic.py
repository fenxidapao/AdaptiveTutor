"""LLM eval 确定性层（PLAN 18.5 D1 + D5 扩黄金集）：rubric 检查器自检 + 报告一致性。

不调真实 LLM（CI 免费常跑）；真实输出质量用 scripts/eval_llm.py --mode live 按需评估。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
from scripts.eval_llm import (  # noqa: E402
    RUBRICS,
    check_diagnostic_schema,
    check_followup_challenges,
    check_hint_no_answer,
    check_no_fabrication,
    check_report_consistency,
    run_check,
)

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden" / "outputs.jsonl"


def test_rubric_golden_all_pass():
    """黄金集驱动：rubric 检查器判定与期望完全一致（防检查器漂移）。"""
    cases = [json.loads(l) for l in GOLDEN.open(encoding="utf-8") if l.strip()]
    passed, fails = run_check(GOLDEN)
    assert fails == [], f"rubric 判定与期望不一致: {fails}"
    assert passed == len(cases)
    # D5 标签纪律：每例必须带 category/agent（失败归因依赖）
    assert all(c.get("category") and c.get("agent") for c in cases), "黄金集案例缺标签"
    # D5 覆盖约束：每个 rubric 检查器都要有黄金样本兜底
    assert {c["rubric"] for c in cases} == set(RUBRICS)


def test_hint_no_answer():
    """诱导泄题断言：方向提示放行，泄题关键词/答案代码形态拦截。"""
    ok, _ = check_hint_no_answer("想想切片的 end 是含还是不含？step 省略时默认是多少？")
    assert ok
    ok, msg = check_hint_no_answer("答案是 a[1:3]，正确代码是 print(a[1:3])。")
    assert not ok and "答案是" in msg
    ok, _ = check_hint_no_answer("在循环里用 for i in range(len(a)): result.append(a[i]) 就对了。")
    assert not ok
    ok, _ = check_hint_no_answer("想想")
    assert not ok  # 过短


def test_followup_challenges():
    """追问反附和：疑问句 + 指向学生内容放行；纯客套/零重叠拦截。"""
    ok, _ = check_followup_challenges({
        "student": "切片就是用冒号取一段元素，比如 a[1:3]。",
        "reply": "冒号取段没错——那 a[1:3] 到底取到第几个？end 含还是不含？"})
    assert ok
    ok, msg = check_followup_challenges({
        "student": "列表切片就是用冒号取一段元素。",
        "reply": "很好的理解！那我们接着学下一个内容吧。"})
    assert not ok and "疑问句" in msg  # 无疑问 → 未形成追问
    ok, msg = check_followup_challenges({
        "student": "字典的键必须是不可变类型，比如字符串、数字、元组。",
        "reply": "明白了！那列表的索引又是什么样的？"})
    assert not ok and "重叠" in msg  # 有疑问但不指向学生内容 → 空转附和


def test_diagnostic_schema():
    """诊断 JSON schema：weak_points 列表（元素带 kp_id）+ avg_correct ∈ [0,1]。"""
    ok, _ = check_diagnostic_schema(
        '{"weak_points": [{"kp_id": "python.list.slice", "reason": "r", "evidence": []}],'
        ' "learning_style": "简答", "avg_correct": 0.4}')
    assert ok
    ok, _ = check_diagnostic_schema('{"weak_points": [], "avg_correct": 0.0}')
    assert ok  # 新用户空薄弱点合法
    ok, _ = check_diagnostic_schema('{"weak_points": []}')  # 缺 avg_correct
    assert not ok
    ok, _ = check_diagnostic_schema('{"weak_points": [], "avg_correct": 1.4}')
    assert not ok  # 越界
    ok, _ = check_diagnostic_schema('{"weak_points": [{"reason": "r"}], "avg_correct": 0.5}')
    assert not ok  # 元素缺 kp_id
    ok, _ = check_diagnostic_schema("这不是 JSON")
    assert not ok


def test_no_fabrication():
    """工具失败反幻觉：诚实兜底放行；编造课程范围外库/答案级代码拦截。"""
    ok, _ = check_no_fabrication("（抱歉，暂时没有该知识点的讲解材料，请换个知识点。）")
    assert ok
    ok, msg = check_no_fabrication("可以直接用 numpy 的切片函数一行搞定。")
    assert not ok and "numpy" in msg
    ok, _ = check_no_fabrication(
        "def solve(): result = [] for i in range(3): result.append(i) return result print(solve())")
    assert not ok  # 答案级代码密度


def test_report_consistency_matches_db(tmp_path):
    """报告前后测数字必须与 assessments 表一致。"""
    dbp = str(tmp_path / "eval.db")
    db.init_db(dbp)
    uid = "u_cons"
    db.get_user(uid, db_path=dbp)
    db.record_assessment(uid, "all", "pretest", "feynman", 0.4, 10, db_path=dbp)
    db.record_assessment(uid, "all", "posttest", "feynman", 0.8, 10, db_path=dbp)

    from agents import assessor
    r = assessor.report(uid, None, dbp)
    ok, msg = check_report_consistency({**r, "user_id": uid}, db, dbp)
    assert ok, msg
    assert r["pre"] == 0.4 and r["post"] == 0.8


def test_report_consistency_catches_drift(tmp_path):
    """篡改库后一致性检查应失败（抓报告与库不一致的回归）。"""
    dbp = str(tmp_path / "drift.db")
    db.init_db(dbp)
    uid = "u_drift"
    db.get_user(uid, db_path=dbp)
    db.record_assessment(uid, "all", "pretest", "feynman", 0.5, 10, db_path=dbp)
    # 模拟"报告与库不一致"：手工改库分数
    import sqlite3
    with sqlite3.connect(dbp) as c:
        c.execute("UPDATE assessments SET score=0.9 WHERE user_id=? AND kind='pretest'", (uid,))
    ok, msg = check_report_consistency({"user_id": uid, "pre": 0.5, "post": None}, db, dbp)
    assert not ok
    assert "pretest" in msg
