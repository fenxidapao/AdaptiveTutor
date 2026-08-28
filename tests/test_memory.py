"""E5 Memory 动态画像测试（PLAN 可扩展点 3 / 9 步框架⑤Memory）：

- db.update_profile_incremental：答错追加薄弱点（带证据链）、重复答错证据去重、
  答对且 mastery>=0.8 移除（薄弱点消除）、无画像首次答错自动建画像；
- feynman.profile_context / generate_followup：历史画像注入（跨会话记忆）。
纯规则，不调真实 LLM（model.chat 打桩）。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
import model  # noqa: E402
from agents import feynman  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    p = str(tmp_path / "state.db")
    db.init_db(p)
    return p


def _weak_ids(user, tmp_db):
    prof = db.get_profile(user, tmp_db)
    return db.parse_weak_ids(prof)


# ==================== 增量更新：答错追加 ====================


def test_wrong_answer_appends_weak_point(tmp_db):
    """答错：不在 weak_points 的 kp 被追加，带 evidence 证据链。"""
    user = "u_mem1"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "ex_a1", "kp.a", False, "错", "x", db_path=tmp_db)
    prof = db.get_profile(user, tmp_db)
    assert prof is not None
    weak = db.parse_weak_details(prof)
    assert len(weak) == 1
    assert weak[0]["kp_id"] == "kp.a"
    assert weak[0]["evidence"] == ["ex_a1"]
    assert "增量" in weak[0]["reason"]


def test_repeated_wrong_evidence_dedup(tmp_db):
    """重复答错：evidence 追加去重，不重复添加 kp。"""
    user = "u_mem2"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "ex_a1", "kp.a", False, "错", "x", db_path=tmp_db)
    db.log_exercise(user, "ex_a1", "kp.a", False, "错", "x", db_path=tmp_db)  # 同题重复
    db.log_exercise(user, "ex_a2", "kp.a", False, "错", "x", db_path=tmp_db)
    weak = db.parse_weak_details(db.get_profile(user, tmp_db))
    assert len([w for w in weak if w["kp_id"] == "kp.a"]) == 1  # kp 只出现一次
    kp_a = next(w for w in weak if w["kp_id"] == "kp.a")
    assert set(kp_a["evidence"]) == {"ex_a1", "ex_a2"}  # 去重


def test_correct_with_mastery_removes_weak_point(tmp_db):
    """答对且 mastery>=0.8：薄弱点移除（动态画像的"消除"）。"""
    user = "u_mem3"
    db.get_user(user, db_path=tmp_db)
    db.save_profile(user, {
        "weak_points": [{"kp_id": "kp.a", "reason": "旧画像", "evidence": ["ex0"]}],
        "learning_style": "代码",
    }, db_path=tmp_db)
    for i in range(4):  # 4 次全对 → mastery=1.0
        db.log_exercise(user, f"ex_ok{i}", "kp.a", True, "对", "ok", db_path=tmp_db)
    assert _weak_ids(user, tmp_db) == []  # 已消除
    # learning_style 保留（增量不覆盖既有偏好）
    prof = db.get_profile(user, tmp_db)
    assert prof["learning_style"] == "代码"


def test_correct_below_mastery_keeps_weak_point(tmp_db):
    """答对但 mastery<0.8：薄弱点保留（单次答对不足证明掌握）。"""
    user = "u_mem4"
    db.get_user(user, db_path=tmp_db)
    db.save_profile(user, {
        "weak_points": [{"kp_id": "kp.a", "reason": "旧画像", "evidence": []}],
        "learning_style": "",
    }, db_path=tmp_db)
    db.log_exercise(user, "e1", "kp.a", False, "错", "x", db_path=tmp_db)  # mastery 0
    db.log_exercise(user, "e2", "kp.a", True, "对", "ok", db_path=tmp_db)  # mastery 0.5
    assert _weak_ids(user, tmp_db) == ["kp.a"]


def test_wrong_answer_creates_profile_when_none(tmp_db):
    """无画像时首次答错：自动创建画像（不用先跑诊断）。"""
    user = "u_mem5"
    db.get_user(user, db_path=tmp_db)
    assert db.get_profile(user, tmp_db) is None
    db.log_exercise(user, "e1", "kp.b", False, "错", "x", db_path=tmp_db)
    assert _weak_ids(user, tmp_db) == ["kp.b"]


def test_correct_without_profile_no_noise(tmp_db):
    """无画像时答对：不创建空画像。"""
    user = "u_mem6"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "e1", "kp.a", True, "对", "ok", db_path=tmp_db)
    assert db.get_profile(user, tmp_db) is None


# ==================== 跨会话记忆：画像注入 ====================


def test_profile_context_summary(tmp_db):
    """profile_context：有画像返回薄弱点+偏好摘要，无画像返回空。"""
    user = "u_mem7"
    db.get_user(user, db_path=tmp_db)
    assert feynman.profile_context(user, tmp_db) == ""
    db.save_profile(user, {
        "weak_points": [{"kp_id": "kp.a", "reason": "", "evidence": []}],
        "learning_style": "代码",
    }, db_path=tmp_db)
    ctx = feynman.profile_context(user, tmp_db)
    assert "kp.a" in ctx
    assert "代码" in ctx


def test_followup_injects_history(monkeypatch, tmp_db):
    """generate_followup：prompt 注入历史画像（跨会话带记忆追问）。"""
    user = "u_mem8"
    db.get_user(user, db_path=tmp_db)
    db.save_profile(user, {
        "weak_points": [{"kp_id": "python.list.slice", "reason": "", "evidence": []}],
        "learning_style": "简答",
    }, db_path=tmp_db)

    captured = {}

    def fake_chat(messages, **kw):
        captured["user"] = messages[-1]["content"]
        return "追问：切片 end 含不含？"

    monkeypatch.setattr(model, "chat", fake_chat)
    kp = {"kp_id": "python.list.slice", "title": "列表切片"}
    feynman.generate_followup(kp, [{"role": "user", "content": "切片就是取子集"}],
                              user, tmp_db)
    assert "历史薄弱点" in captured["user"]
    assert "python.list.slice" in captured["user"]
    assert "简答" in captured["user"]


def test_followup_without_user_no_injection(monkeypatch, tmp_db):
    """不带 user_id 调用：不注入画像（兼容既有无状态调用）。"""
    captured = {}

    def fake_chat(messages, **kw):
        captured["user"] = messages[-1]["content"]
        return "追问一下"

    monkeypatch.setattr(model, "chat", fake_chat)
    kp = {"kp_id": "python.list.slice", "title": "列表切片"}
    feynman.generate_followup(kp, [{"role": "user", "content": "讲解"}])
    assert "历史薄弱点" not in captured["user"]
