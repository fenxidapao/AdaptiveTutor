"""E 阶段 Loop 工程化测试（PLAN 20）：回流判定/状态机、策略切换、学习队列。

纯规则测试，不依赖 LLM（防漂移是本模块设计核心）：
- E1 needs_reflow 各分支 + reflow_after_posttest 完整状态机流转；
- E2 practice_strategy 连续失败 0/1/2/3+ 次的策略降级；
- E3 daily_queue 到期优先 + mastery 升序 + limit 截断 + 未学不进队列。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import db  # noqa: E402
from agents import loop  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    p = str(tmp_path / "state.db")
    db.init_db(p)
    return p


# ==================== E1：needs_reflow 判定 ====================


def test_needs_reflow_above_threshold():
    """后测 >= 阈值：不回流。"""
    assert loop.needs_reflow(0.2, 0.6) is False
    assert loop.needs_reflow(0.2, 0.9) is False


def test_needs_reflow_no_pretest():
    """无前测基线：只看后测。"""
    assert loop.needs_reflow(None, 0.4) is True
    assert loop.needs_reflow(None, 0.7) is False


def test_needs_reflow_insufficient_gain():
    """后测不及格且提升不足（<0.3）：回流。"""
    assert loop.needs_reflow(0.3, 0.5) is True   # +0.2 不足，回流
    assert loop.needs_reflow(0.4, 0.55) is True  # +0.15 不足，回流


def test_needs_reflow_significant_gain():
    """起点低但进步显著（>=0.3）：不强留回流。"""
    assert loop.needs_reflow(0.1, 0.5) is False  # 0.5<0.6 但 +0.4
    assert loop.needs_reflow(0.0, 0.4) is False  # 0.4<0.6 但 +0.4
    assert loop.needs_reflow(0.3, 0.55) is True  # +0.25 < 0.3 且 0.55 < 0.6


# ==================== E1：reflow_after_posttest 状态机 ====================


def test_reflow_not_triggered_when_passing(tmp_db):
    """初次后测达标（>=0.6 或提升显著）：不创建回流。"""
    user = "u_loop1"
    db.get_user(user, db_path=tmp_db)
    r = loop.reflow_after_posttest(user, "ch1", 0.8, ["kp.a"], tmp_db)
    assert r["triggered"] is False
    assert db.list_reflows(user, tmp_db) == []


def test_reflow_round1_open_on_failing(tmp_db):
    """初次后测不达标：创建第 1 轮回流任务（open），weak_kps 落库。"""
    user = "u_loop2"
    db.get_user(user, db_path=tmp_db)
    db.record_assessment(user, "ch1", "pretest", "feynman", 0.2, 10, db_path=tmp_db)
    r = loop.reflow_after_posttest(user, "ch1", 0.4, ["kp.a", "kp.b"], tmp_db)
    assert r["triggered"] is True
    assert r["passed"] is None
    assert r["round"] == 1
    reflows = db.list_reflows(user, tmp_db)
    assert len(reflows) == 1
    assert reflows[0]["status"] == "open"
    assert "kp.a" in reflows[0]["weak_kps"]


def test_reflow_completed_on_pass_retest(tmp_db):
    """重测达标（>=0.8）：闭环完成。"""
    user = "u_loop3"
    db.get_user(user, db_path=tmp_db)
    db.record_assessment(user, "ch1", "pretest", "feynman", 0.2, 10, db_path=tmp_db)
    loop.reflow_after_posttest(user, "ch1", 0.4, ["kp.a"], tmp_db)
    r = loop.reflow_after_posttest(user, "ch1", 0.9, [], tmp_db)
    assert r["passed"] is True
    assert db.list_reflows(user, tmp_db)[0]["status"] == "completed"
    assert r["reflow"]["retest_score"] == 0.9


def test_reflow_next_round_on_fail_retest(tmp_db):
    """重测不达标且未超轮：本轮 failed，开第 2 轮。"""
    user = "u_loop4"
    db.get_user(user, db_path=tmp_db)
    loop.reflow_after_posttest(user, "ch1", 0.5, ["kp.a"], tmp_db)
    r = loop.reflow_after_posttest(user, "ch1", 0.6, ["kp.a"], tmp_db)
    # 0.6 < 0.8 不达标，round 1 -> failed，round 2 open
    assert r["passed"] is False
    assert r["round"] == 2
    reflows = db.list_reflows(user, tmp_db)
    assert [x["status"] for x in reflows] == ["failed", "open"]


def test_reflow_given_up_after_max_rounds(tmp_db):
    """超轮（round >= 上限）仍不达标：given_up，不再开新轮。"""
    user = "u_loop5"
    db.get_user(user, db_path=tmp_db)
    loop.reflow_after_posttest(user, "ch1", 0.5, ["kp.a"], tmp_db)   # round1 open
    loop.reflow_after_posttest(user, "ch1", 0.5, ["kp.a"], tmp_db)   # round1 failed, round2 open
    r = loop.reflow_after_posttest(user, "ch1", 0.5, ["kp.a"], tmp_db)  # round2 超轮 → given_up
    assert r["gave_up"] is True
    statuses = [x["status"] for x in db.list_reflows(user, tmp_db)]
    assert statuses == ["failed", "given_up"]
    assert db.get_open_reflow(user, "ch1", tmp_db) is None


def test_reflow_status_active_and_weak_kps(tmp_db):
    """reflow_status：open 时返回活动回流与薄弱点清单。"""
    user = "u_loop6"
    db.get_user(user, db_path=tmp_db)
    loop.reflow_after_posttest(user, "ch1", 0.4, ["kp.a"], tmp_db)
    s = loop.reflow_status(user, "ch1", tmp_db)
    assert s["active"] is True
    assert s["round"] == 1
    assert s["weak_kps"] == ["kp.a"]
    assert s["max_rounds"] == config.REFLOW_MAX_ROUNDS


# ==================== E2：练习策略切换 ====================


def test_strategy_no_record_or_recent_pass(tmp_db):
    """无记录 / 最近答对：hint。"""
    user = "u_strat"
    db.get_user(user, db_path=tmp_db)
    assert loop.practice_strategy(user, "python.list.slice", tmp_db) == "hint"
    db.log_exercise(user, "e1", "python.list.slice", True, "ok", "对", db_path=tmp_db)
    assert loop.practice_strategy(user, "python.list.slice", tmp_db) == "hint"


def test_strategy_one_fail_hint(tmp_db):
    """连续失败 1 次：hint。"""
    user = "u_strat1"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "e1", "python.list.slice", False, "错", "不对", db_path=tmp_db)
    assert loop.practice_strategy(user, "python.list.slice", tmp_db) == "hint"


def test_strategy_two_fails_explain(tmp_db):
    """连续失败 2 次：降级为 explain（标准讲解+对比举例）。"""
    user = "u_strat2"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "e1", "python.list.slice", False, "错", "x", db_path=tmp_db)
    db.log_exercise(user, "e2", "python.list.slice", False, "错", "x", db_path=tmp_db)
    assert loop.practice_strategy(user, "python.list.slice", tmp_db) == "explain"


def test_strategy_three_fails_prereq(tmp_db):
    """连续失败 3 次：prereq（建议复习前置知识点）。"""
    user = "u_strat3"
    db.get_user(user, db_path=tmp_db)
    for i in range(3):
        db.log_exercise(user, f"e{i}", "python.list.slice", False, "错", "x",
                        db_path=tmp_db)
    assert loop.practice_strategy(user, "python.list.slice", tmp_db) == "prereq"


def test_strategy_fail_then_pass_resets(tmp_db):
    """失败后再答对：策略回到 hint（连续失败被打破）。"""
    user = "u_strat4"
    db.get_user(user, db_path=tmp_db)
    for i in range(2):
        db.log_exercise(user, f"e{i}", "kp.x", False, "错", "x", db_path=tmp_db)
    db.log_exercise(user, "e2", "kp.x", True, "对", "ok", db_path=tmp_db)
    assert loop.practice_strategy(user, "kp.x", tmp_db) == "hint"


def test_prereq_titles_from_graph(tmp_db):
    """prereq_titles：从知识图谱 prerequisites 取前置标题。"""
    titles = loop.prereq_titles("python.list.slice")
    assert isinstance(titles, list)
    assert titles  # 图谱中 slice 必有前置
    assert all("python" in t for t in titles)  # 前置带 kp_id


# ==================== E3：掌握度驱动学习队列 ====================


def test_queue_due_review_first(tmp_db):
    """到期复习优先于未掌握知识点。"""
    user = "u_q1"
    db.get_user(user, db_path=tmp_db)
    today = datetime.now().date().isoformat()
    # 到期复习的 kp
    db.upsert_kp(user, {"kp_id": "python.list.slice", "title": "切片",
                        "chapter": "ch2", "next_review": today},
                 db_path=tmp_db)
    # 已学未掌握（答错 → mastery=0）
    db.log_exercise(user, "e1", "python.variable.assignment", False, "错", "x",
                    db_path=tmp_db)
    q = loop.daily_queue(user, "python", db_path=tmp_db)
    assert q[0]["kp_id"] == "python.list.slice"
    assert q[0]["reason"] == "review"
    # 未掌握的知识点在后面
    rest = [x["kp_id"] for x in q[1:]]
    assert "python.variable.assignment" in rest


def test_queue_weak_sorted_by_mastery_asc(tmp_db):
    """已学未掌握：按 mastery 升序（最弱先学）。"""
    user = "u_q2"
    db.get_user(user, db_path=tmp_db)
    # kp_a 全对 → mastery 1.0（不弱）；kp_b 1对1错 → 0.5；kp_c 全错 → 0
    db.log_exercise(user, "a1", "kp_a", True, "ok", "x", db_path=tmp_db)
    db.log_exercise(user, "b1", "kp_b", True, "ok", "x", db_path=tmp_db)
    db.log_exercise(user, "b2", "kp_b", False, "错", "x", db_path=tmp_db)
    db.log_exercise(user, "c1", "kp_c", False, "错", "x", db_path=tmp_db)
    q = loop.daily_queue(user, "python", db_path=tmp_db)
    weak_ids = [x["kp_id"] for x in q if x["reason"] == "weak"]
    # kp_c(0) 在 kp_b(0.5) 前面；kp_a(mastery=1.0) 不进队列
    assert weak_ids.index("kp_c") < weak_ids.index("kp_b")
    assert "kp_a" not in weak_ids


def test_queue_unseen_not_included(tmp_db):
    """没学过的知识点（seen=0）不进队列。"""
    user = "u_q3"
    db.get_user(user, db_path=tmp_db)
    q = loop.daily_queue(user, "python", db_path=tmp_db)
    assert q == []  # 无到期、无已学未掌握


def test_queue_limit(tmp_db):
    """limit 截断。"""
    user = "u_q4"
    db.get_user(user, db_path=tmp_db)
    for i in range(6):
        db.log_exercise(user, f"e{i}", f"kp_w{i}", False, "错", "x", db_path=tmp_db)
    q = loop.daily_queue(user, "python", limit=3, db_path=tmp_db)
    assert len(q) == 3


def test_queue_due_also_weak_not_duplicated(tmp_db):
    """到期且未掌握的知识点只出现一次（reason=review 优先）。"""
    user = "u_q5"
    db.get_user(user, db_path=tmp_db)
    today = datetime.now().date().isoformat()
    db.upsert_kp(user, {"kp_id": "python.list.slice", "title": "切片",
                        "chapter": "ch2", "next_review": today},
                 db_path=tmp_db)
    db.log_exercise(user, "e1", "python.list.slice", False, "错", "x", db_path=tmp_db)
    q = loop.daily_queue(user, "python", db_path=tmp_db)
    ids = [x["kp_id"] for x in q]
    assert ids.count("python.list.slice") == 1
