"""C2 开闸前修复回归（2026-08-29，开闸自查发现的问题）。

覆盖：
1. P0-1：/api/feynman/* 按 group_name 硬门禁——lecture 组 403（自变量强制，
   前端分支只是 UX，测量有效性靠服务端）；
2. P0-2：/api/users* 三端点任何模式下不泄露 password_hash + 实验模式鉴权；
3. P2：缺 user_id/ex_id/course/kp_id 的畸形请求 400（原 500 KeyError）；
4. P2：_auto_group 只统计有凭据账号（无凭据遗留账号不占均衡计数）。
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import db  # noqa: E402
import model  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402

client = TestClient(app)


def new_uid():
    return f"u_c2_{uuid.uuid4().hex[:8]}"


def _register(group=None):
    """注册一个用户返回 (uid, session_id, group_name)；group 传入时强制改组。"""
    uid = new_uid()
    r = client.post("/api/register", json={"user_id": uid, "password": "pass123"})
    assert r.status_code == 200
    user = r.json()["user"]
    if group and user["group_name"] != group:
        with db._conn() as c:
            c.execute("UPDATE users SET group_name=? WHERE user_id=?", (group, uid))
    return uid, r.json()["session_id"], group or user["group_name"]


# ==================== P0-2：用户端点泄露 ====================

def test_users_endpoints_never_leak_password_hash(monkeypatch):
    """三种模式下 /api/users* 的任何响应都不得含 password_hash。"""
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", False)  # 演示模式也要剥
    uid = new_uid()
    client.post("/api/register", json={"user_id": uid, "password": "pass123"})

    r_list = client.get("/api/users")
    assert r_list.status_code == 200
    assert all("password_hash" not in u for u in r_list.json())

    r_one = client.get(f"/api/users/{uid}")
    assert r_one.status_code == 200
    assert "password_hash" not in r_one.json()

    r_up = client.post("/api/users", params={"user_id": new_uid(), "name": "x"})
    assert r_up.status_code == 200
    assert "password_hash" not in r_up.json()


def test_users_endpoints_require_auth_in_experiment_mode(monkeypatch):
    """EXPERIMENT_AUTH=1：列表/单查需会话，POST /api/users 禁用（走 /api/register）。"""
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", True)
    uid, sid, _ = _register()

    assert client.get("/api/users").status_code == 401  # 未登录
    assert client.get("/api/users", params={"session_id": sid}).status_code == 200

    assert client.get(f"/api/users/{uid}").status_code == 401  # 未登录
    assert client.get(f"/api/users/{uid}", params={"session_id": sid}).status_code == 200
    # 会话与目标用户不一致 → 403
    other, _, _ = _register()
    r_forge = client.get(f"/api/users/{other}", params={"session_id": sid})
    assert r_forge.status_code == 403

    r_legacy = client.post("/api/users", params={"user_id": new_uid()})
    assert r_legacy.status_code == 403  # 绕过密码注册在实验模式下关闭


# ==================== P0-1：费曼端点分组硬门禁 ====================

def test_feynman_endpoints_blocked_for_lecture(monkeypatch):
    """lecture 组调费曼端点 → 403（实验自变量强制）。"""
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", True)
    uid, sid, _ = _register(group="lecture")
    body = {"course": "python", "kp_id": "python.list.slice",
            "user_id": uid, "session_id": sid, "transcript": [
                {"role": "user", "content": "切片就是用冒号取一段。"}]}
    assert client.post("/api/feynman/turn", json=body).status_code == 403
    assert client.post("/api/feynman/summarize", json=body).status_code == 403


def test_feynman_endpoints_allowed_for_feynman_group(monkeypatch):
    """feynman 组正常走追问（mock LLM）→ 200。"""
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", True)
    monkeypatch.setattr(model, "chat", lambda *a, **kw: "那 end 含还是不含？")
    uid, sid, _ = _register(group="feynman")
    r = client.post("/api/feynman/turn", json={
        "course": "python", "kp_id": "python.list.slice",
        "user_id": uid, "session_id": sid,
        "transcript": [{"role": "user", "content": "切片就是用冒号取一段元素。"}]})
    assert r.status_code == 200
    assert r.json()["coach"]


# ==================== P2：畸形请求 400 ====================

def test_malformed_requests_return_400(monkeypatch):
    """缺 user_id/answers/ex_id → 400（原实现 KeyError → 500）。"""
    monkeypatch.setattr(config, "EXPERIMENT_AUTH", False)
    r1 = client.post("/api/quiz/python/pretest/submit", json={})
    assert r1.status_code == 400
    r2 = client.post("/api/grade", json={"user_id": "u_x"})
    assert r2.status_code == 400
    # feynman 端点缺字段走兜底默认值 → 404"未知知识点: 缺失"，不再是 500 KeyError
    r3 = client.post("/api/feynman/turn", json={"course": "python"})
    assert r3.status_code == 404


# ==================== P2：分组均衡计数 ====================

def test_auto_group_ignores_passwordless_users():
    """无凭据遗留账号（password_hash NULL）不应翻转下一次分组的均衡方向。"""
    g0 = db._auto_group()
    ghost = f"u_c2_ghost_{uuid.uuid4().hex[:6]}"
    db.get_user(ghost)  # 隐式建号：无 password_hash
    with db._conn() as c:
        c.execute("UPDATE users SET group_name=? WHERE user_id=?", (g0, ghost))
    try:
        assert db._auto_group() == g0  # 幽灵账号不占用计数
    finally:
        with db._conn() as c:
            c.execute("DELETE FROM users WHERE user_id=?", (ghost,))


# ==================== C2 运维：前端缓存 ====================

def test_frontend_no_cache_headers():
    """/ 与 /static/* 必须带 Cache-Control: no-cache——实验期间热修前端
    若被浏览器启发式缓存，学生会拿旧 JS 做题，实验条件被静默撕裂。"""
    for path in ("/", "/static/app.js"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers.get("cache-control") == "no-cache", f"{path} 缺 no-cache"


# ==================== C2 运维：报告柱状图数据 ====================

def test_report_by_chapter_synthesized_for_web_flow():
    """Web 实验流测评 chapter 恒为 'all'，assessor.by_chapter 会排除它——
    API 层必须合成聚合行，否则报告柱状图永远为空（2026-08-29 手机实测发现）。"""
    uid = new_uid()
    db.get_user(uid)
    db.record_assessment(uid, "all", "pretest", "feynman", 0.4, 10)
    db.record_assessment(uid, "all", "posttest", "feynman", 0.8, 10)
    r = client.get(f"/api/report/python/{uid}")
    assert r.status_code == 200
    body = r.json()
    assert body["pre"] == 0.4 and body["post"] == 0.8
    assert len(body["by_chapter"]) == 1
    assert body["by_chapter"][0]["chapter"] == "all"
    assert body["by_chapter"][0]["pre"] == 0.4 and body["by_chapter"][0]["post"] == 0.8
