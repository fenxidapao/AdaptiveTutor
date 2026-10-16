"""C1 鉴权/配额/mode 强制测试（2026-08-23）。

覆盖：
1. db 层：注册/密码校验/session 生命周期/自动分组均衡/配额计数
2. API 层：register/login/me/logout
3. EXPERIMENT_AUTH=1 时：未登录 401、mode 按 group_name 服务端强制
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import db  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402

client = TestClient(app)


def new_uid():
    return f"u_auth_{uuid.uuid4().hex[:8]}"


# ==================== db 层 ====================

def test_register_and_verify():
    uid = new_uid()
    user = db.register_user(uid, "pass123", name="测试")
    assert user["user_id"] == uid
    assert user["group_name"] in ("feynman", "lecture")  # 自动分组
    assert user["password_hash"] and "$" in user["password_hash"]  # salt$hash
    assert db.verify_user(uid, "pass123") is True
    assert db.verify_user(uid, "wrong") is False
    with pytest.raises(ValueError):
        db.register_user(uid, "pass123")  # 重复注册


def test_session_lifecycle():
    uid = new_uid()
    db.register_user(uid, "pass123")
    sid = db.create_session(uid)
    assert len(sid) == 32  # uuid4 hex
    assert db.get_session_user(sid) == uid
    db.delete_session(sid)
    assert db.get_session_user(sid) is None


def test_auto_group_balance():
    """注册均衡分配：两组人数差不超过 1。"""
    groups = []
    for _ in range(4):
        u = db.register_user(new_uid(), "pass123")
        groups.append(u["group_name"])
    assert abs(groups.count("feynman") - groups.count("lecture")) <= 1


def test_quota_count():
    uid = new_uid()
    db.register_user(uid, "pass123")
    assert db.today_usage(uid) == 0
    db.log_llm_call("test", "m", 10, 10, 20, 5, user_id=uid, session_id="sess_abc")
    assert db.today_usage(uid) == 1
    exceeded, _ = db.quota_exceeded(uid)
    assert exceeded is False  # 1 < 50 上限
    # D4 session trace：session_id 落库可回放
    with db._conn() as c:
        row = c.execute("SELECT session_id FROM llm_logs ORDER BY id DESC LIMIT 1").fetchone()
    assert row["session_id"] == "sess_abc"


# ==================== API 层 ====================

def test_api_register_login_me():
    uid = new_uid()
    r = client.post("/api/register", json={"user_id": uid, "password": "pass123"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["user"]["group_name"] in ("feynman", "lecture")
    assert "password_hash" not in r.json()["user"]  # 密码哈希绝不回传前端

    # 重复注册 → 409
    r2 = client.post("/api/register", json={"user_id": uid, "password": "pass456"})
    assert r2.status_code == 409

    # me（有效 session）
    r3 = client.get(f"/api/me?session_id={sid}")
    assert r3.status_code == 200
    assert r3.json()["user_id"] == uid

    # 登录错密码 → 401
    r4 = client.post("/api/login", json={"user_id": uid, "password": "bad"})
    assert r4.status_code == 401

    # 退出后 me → 401
    client.post("/api/logout", json={"session_id": sid})
    r5 = client.get(f"/api/me?session_id={sid}")
    assert r5.status_code == 401


def test_api_register_requires_password():
    r = client.post("/api/register", json={"user_id": new_uid(), "password": ""})
    assert r.status_code == 400
    r = client.post("/api/register", json={"user_id": new_uid(), "password": "123"})
    assert r.status_code == 400  # 密码太短


# ==================== 实验模式（EXPERIMENT_AUTH=1） ====================

def test_experiment_mode_forced(monkeypatch):
    """开启实验模式后：
    - 未登录访问业务端点 → 401
    - 提交前后测 mode 由服务端按 group_name 强制（前端传的 mode 无效）
    """
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", True)
    uid = new_uid()
    r = client.post("/api/register", json={"user_id": uid, "password": "pass123"})
    sid = r.json()["session_id"]
    group = r.json()["user"]["group_name"]

    # 未登录 → 401
    r401 = client.post("/api/diagnose/u0")
    assert r401.status_code == 401
    r401b = client.post("/api/quiz/python/pretest/submit",
                        json={"user_id": uid, "answers": {}})
    assert r401b.status_code == 401

    # 带 session 提交前测：mode 强制为 group，前端传 feynman 无效
    r_sub = client.post("/api/quiz/python/pretest/submit", json={
        "user_id": uid, "session_id": sid, "answers": {}, "mode": "feynman",
    })
    assert r_sub.status_code == 200
    import learning_pack
    n = len(learning_pack.load_pretest("python"))
    assert r_sub.json()["total"] == n
    rows = db.list_assessments(uid)
    assert rows[-1]["mode"] == group  # 服务端强制，不是前端传的 feynman

    # 会话伪装他人 → 403
    other = new_uid()
    client.post("/api/register", json={"user_id": other, "password": "pass123"})
    r_forge = client.post("/api/quiz/python/pretest/submit", json={
        "user_id": other, "session_id": sid, "answers": {},
    })
    assert r_forge.status_code == 403
