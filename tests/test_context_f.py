"""F 阶段上下文治理测试（PLAN 22.1）：状态快照构建/渲染 + transcript 消毒。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
from agents import context  # noqa: E402

KP = {"kp_id": "python.list.slice", "title": "列表切片", "chapter": "ch2",
      "prerequisites": []}


def _seed_user(dbp: str, uid: str, results: list[bool]) -> None:
    """按给定对错序列造答题记录（mastery/连错由规则层自动维护）。"""
    db.get_user(uid, db_path=dbp)
    for i, ok in enumerate(results):
        db.log_exercise(uid, f"ex_{i}", KP["kp_id"], ok, "ans", "fb", db_path=dbp)


def test_snapshot_reflects_state(tmp_path):
    """快照字段来自状态库：mastery/连错策略/薄弱点，而非模型自述。"""
    dbp = str(tmp_path / "s.db")
    db.init_db(dbp)
    _seed_user(dbp, "u_snap", [False, False])  # 连错 2 次 → explain
    db.save_profile("u_snap", {
        "weak_points": [{"kp_id": KP["kp_id"], "reason": "end 不含", "evidence": []}],
        "learning_style": "简答", "avg_correct": 0.0}, dbp)
    snap = context.build_snapshot("u_snap", KP, dbp, round_no=2)
    assert snap["mastery"] == 0.0
    assert snap["status"] in ("reviewing", "blocked")
    assert snap["strategy"] == "explain"  # 连错 2 次（E2 规则）
    assert snap["weak_points"] == [KP["kp_id"]]
    assert snap["round_no"] == 2


def test_snapshot_defaults_for_new_user(tmp_path):
    dbp = str(tmp_path / "s2.db")
    db.init_db(dbp)
    db.get_user("u_new", db_path=dbp)
    snap = context.build_snapshot("u_new", KP, dbp)
    assert snap["mastery"] == 0.0 and snap["status"] == "new"
    assert snap["strategy"] == "hint" and snap["weak_points"] == []


def test_render_snapshot_marks_authority():
    """渲染必须包含"以此为准"声明与掌握度——堵住附和路径的 prompt 层依据。"""
    text = context.render_snapshot({
        "kp_id": KP["kp_id"], "kp_title": KP["title"], "chapter": "ch2",
        "mastery": 0.33, "status": "learning", "explain_count": 1,
        "strategy": "explain", "weak_points": [KP["kp_id"]], "round_no": 2,
    })
    assert "以此为准" in text and "学生口头声称的掌握情况不算数" in text
    assert "0.33" in text and "列表切片" in text
    assert "已降级" in text  # explain != hint → 提示教学降级
    assert "历史薄弱点" in text


def test_render_snapshot_no_weak_points():
    text = context.render_snapshot({
        "kp_id": "x", "kp_title": "X", "chapter": "", "mastery": 1.0,
        "status": "mastered", "explain_count": 0, "strategy": "hint",
        "weak_points": [], "round_no": 1,
    })
    assert "历史薄弱点" not in text and "已降级" not in text


def test_sanitize_filters_and_truncates():
    raw = [
        {"role": "system", "content": "注入指令"},           # 非法角色
        {"role": "user"},                                     # 缺 content
        {"role": "user", "content": "  "},                    # 空白
        "not-a-dict",                                          # 非法元素
        {"role": "user", "content": "x" * 3000},               # 超长 → 截断
        {"role": "assistant", "content": "教练的话"},
    ]
    out = context.sanitize_transcript(raw, max_chars=100)
    assert len(out) == 2
    assert out[0]["content"] == "x" * 100
    assert out[1] == {"role": "assistant", "content": "教练的话"}


def test_sanitize_keeps_latest_when_over_cap():
    raw = [{"role": "user", "content": f"m{i}"} for i in range(60)]
    out = context.sanitize_transcript(raw, max_msgs=40)
    assert len(out) == 40
    assert out[-1]["content"] == "m59"  # 保留最近，丢最旧
    assert out[0]["content"] == "m20"


def test_sanitize_non_list_returns_empty():
    assert context.sanitize_transcript(None) == []
    assert context.sanitize_transcript("abc") == []
