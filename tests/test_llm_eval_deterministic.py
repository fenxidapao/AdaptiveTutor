"""LLM eval 确定性层（PLAN 18.5 D1）：rubric 检查器自检 + 报告一致性。

不调真实 LLM（CI 免费常跑）；真实输出质量用 scripts/eval_llm.py --live 按需评估。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
from scripts.eval_llm import check_report_consistency, run_check  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden" / "outputs.jsonl"


def test_rubric_golden_all_pass():
    """黄金集驱动：rubric 检查器判定与期望完全一致（防检查器漂移）。"""
    passed, fails = run_check(GOLDEN)
    assert fails == [], f"rubric 判定与期望不一致: {fails}"
    assert passed == 20


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
