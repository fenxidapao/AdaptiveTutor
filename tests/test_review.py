"""间隔复习调度测试（P2）：SM-2 简化算法的核心规则。

纯规则测试，不依赖 LLM：间隔翻倍/答错重置/到期筛选/档位存档。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
from agents import scheduler  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    p = str(tmp_path / "state.db")
    db.init_db(p)
    return p


def _init_kp(user, kp_id, tmp_db, next_review=None, interval=0):
    db.upsert_kp(user, {"kp_id": kp_id, "title": kp_id, "chapter": "ch2"},
                 db_path=tmp_db)
    if next_review:
        db.upsert_kp(user, {"kp_id": kp_id, "next_review": next_review},
                     db_path=tmp_db)
    with db._conn(tmp_db) as c:
        c.execute("UPDATE knowledge_points SET review_interval=? "
                  "WHERE user_id=? AND kp_id=?", (interval, user, kp_id))


def test_interval_doubles_on_correct(tmp_db):
    """答对：间隔 1→2→4 翻倍，review_interval 档位递增。"""
    user, kp = "u_rev", "python.list.slice"
    db.get_user(user, db_path=tmp_db)
    _init_kp(user, kp, tmp_db, next_review="2026-01-01")

    d1 = scheduler.schedule_next(user, kp, True, tmp_db)
    assert d1 == 1  # 首次答对 → 1 天
    k = db.get_kp(user, kp, tmp_db)
    assert k["review_interval"] == 1

    d2 = scheduler.schedule_next(user, kp, True, tmp_db)
    assert d2 == 2
    d3 = scheduler.schedule_next(user, kp, True, tmp_db)
    assert d3 == 4


def test_interval_resets_on_wrong(tmp_db):
    """答错：重置为 1 天，档位归零。"""
    user, kp = "u_rev2", "python.dict.basic"
    db.get_user(user, db_path=tmp_db)
    _init_kp(user, kp, tmp_db, next_review="2026-01-01", interval=3)

    d = scheduler.schedule_next(user, kp, False, tmp_db)
    assert d == 1
    k = db.get_kp(user, kp, tmp_db)
    assert k["review_interval"] == 0


def test_interval_caps_at_30(tmp_db):
    """答对多次后封顶 30 天，不无限翻倍。"""
    user, kp = "u_rev3", "python.list.comprehension"
    db.get_user(user, db_path=tmp_db)
    _init_kp(user, kp, tmp_db, next_review="2026-01-01", interval=5)

    d = scheduler.schedule_next(user, kp, True, tmp_db)
    assert d == 30
    d2 = scheduler.schedule_next(user, kp, True, tmp_db)
    assert d2 == 30


def test_due_kps_filters(tmp_db):
    """到期筛选：过期/今天的到期，未来的不到期。"""
    user = "u_due"
    db.get_user(user, db_path=tmp_db)
    today = datetime.now().date().isoformat()
    past = (datetime.now() - timedelta(days=2)).date().isoformat()
    future = (datetime.now() + timedelta(days=3)).date().isoformat()
    _init_kp(user, "kp.past", tmp_db, next_review=past)
    _init_kp(user, "kp.today", tmp_db, next_review=today)
    _init_kp(user, "kp.future", tmp_db, next_review=future)
    due = scheduler.due_kps(user, db_path=tmp_db)
    ids = {k["kp_id"] for k in due}
    assert ids == {"kp.past", "kp.today"}


def test_review_session_updates_mastery_and_interval(tmp_db, monkeypatch):
    """完整复习一轮：答对 → mastery 更新 + 间隔翻倍 + 状态 mastered。"""
    from tests.verify_learning_pack import SAMPLES
    user = "u_session"
    db.get_user(user, db_path=tmp_db)
    kp_id = "python.variable.assignment"
    _init_kp(user, kp_id, tmp_db, next_review="2026-01-01")
    # 复习用该 kp 的练习题，全喂标准答案（答对）
    exs = scheduler.exercises_for_review(kp_id)
    answers = [SAMPLES.get(e["ex_id"], "print(1)") for e in exs]
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))

    res = scheduler.run_review_session(user, "python", ask_user=lambda p: next(it),
                                       db_path=tmp_db)
    assert res["reviewed"] == 1
    assert res["correct"] == 1
    kp = db.get_kp(user, kp_id, tmp_db)
    assert kp["mastery"] == 1.0
    assert kp["review_interval"] == 1
    assert kp["status"] == "mastered"


def test_review_session_empty_when_none_due(tmp_db):
    """无到期知识点：返回空统计，不抛错。"""
    user = "u_empty"
    db.get_user(user, db_path=tmp_db)
    res = scheduler.run_review_session(user, "python", ask_user=lambda p: "",
                                       db_path=tmp_db)
    assert res["reviewed"] == 0
