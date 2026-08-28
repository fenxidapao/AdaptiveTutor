"""Web API 测试（P3）：FastAPI TestClient 验证关键端点（不依赖真实 LLM/服务）。

真实 LLM 调用会慢且花钱，这里只测规则链路（学习包/判题/报告/热力图/用户），
费曼/讲解等 LLM 端点用 monkeypatch 打桩。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture()
def clean_user():
    """每次测试用独立用户，避免数据污染。"""
    import db
    uid = f"u_api_{abs(hash(str(pytest)))}"[0]  # 简化：固定测试用户
    return "u_api"


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "FeynmanTutor" in r.text


def test_learning_pack():
    r = client.get("/api/learning-pack/python")
    assert r.status_code == 200
    assert len(r.json()["knowledge_points"]) == 12


def test_quiz_no_answer_leak():
    """取题接口必须不含答案（防作弊）。"""
    for kind in ("pretest", "posttest"):
        r = client.get(f"/api/quiz/python/{kind}")
        assert r.status_code == 200
        for q in r.json():
            assert "answer" not in q
            assert "check" not in q
            assert "expect_stdout" not in q
            assert "tests" not in q


def test_quiz_submit_scores(clean_user):
    """提交前测：全对 → score=1.0，且写入 assessments。"""
    import db
    import learning_pack

    user = clean_user
    db.get_user(user)
    # 清旧数据
    with db._conn() as c:
        for t in ("assessments", "exercise_logs", "knowledge_points", "profile"):
            c.execute(f"DELETE FROM {t} WHERE user_id=?", (user,))
    pre = learning_pack.load_pretest()
    answers = {q["ex_id"]: str(q["check"]["answer"]) for q in pre if q["type"] == "mcq"}
    r = client.post("/api/quiz/python/pretest/submit",
                    json={"user_id": user, "answers": answers, "mode": "feynman"})
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 1.0
    assert body["correct"] == body["total"]
    assert body["total"] == 10
    ass = db.get_assessments(user)
    assert any(a["kind"] == "pretest" for a in ass)


def test_grade_updates_mastery(clean_user):
    """判题后 knowledge_points.mastery 必须更新（P3 热力图依赖）。"""
    import db
    import learning_pack

    user = clean_user
    db.get_user(user)
    with db._conn() as c:
        for t in ("assessments", "exercise_logs", "knowledge_points", "profile"):
            c.execute(f"DELETE FROM {t} WHERE user_id=?", (user,))
    # 找到一道 output 题，用标准答案判对
    ex = next(e for e in learning_pack.load_exercises()
              if e["type"] == "output" and e["kp_id"] == "python.variable.assignment")
    # 从 verify_learning_pack 借标准答案
    from tests.verify_learning_pack import SAMPLES
    sample = SAMPLES[ex["ex_id"]]
    r = client.post("/api/grade",
                    json={"user_id": user, "ex_id": ex["ex_id"], "answer": sample})
    assert r.status_code == 200
    assert r.json()["correct"] is True
    kp = db.get_kp(user, ex["kp_id"])
    assert kp is not None
    assert kp["mastery"] == 1.0
    assert kp["status"] == "mastered"


def test_heatmap_returns_12_cells():
    r = client.get("/api/heatmap/u0")
    assert r.status_code == 200
    cells = r.json()["cells"]
    assert len(cells) == 12
    assert all("mastery" in c and "kp_id" in c for c in cells)


def test_explain_notes_fallback(monkeypatch, clean_user):
    """RAG 挂掉时讲解走 notes 兜底（打桩避免真实调用）。"""
    import rag
    from agents import feynman

    def boom(*a, **k):
        raise rag.RAGError("离线")

    monkeypatch.setattr(rag, "_retrieve_raw", boom)
    r = client.get("/api/explain/python/python.list.slice", params={"user_id": clean_user})
    assert r.status_code == 200
    assert "切片" in r.json()["explanation"]


def test_path_and_recommend(clean_user):
    """路径/推荐：规则链路不依赖 LLM 也能返回。"""
    import db
    user = clean_user
    db.get_user(user)
    r1 = client.get(f"/api/path/python/{user}")
    assert r1.status_code == 200
    assert len(r1.json()["path"]) == 12
    r2 = client.get(f"/api/recommend/python/{user}", params={"top_n": 3})
    assert r2.status_code == 200
    assert len(r2.json()["recommendations"]) <= 3


def test_register_and_group():
    import db
    uid = "u_api_group"
    r = client.post("/api/users", params={"user_id": uid, "name": "分组测试"})
    assert r.status_code == 200
    assert r.json()["name"] == "分组测试"
    with db._conn() as c:
        c.execute("UPDATE users SET group_name='feynman' WHERE user_id=?", (uid,))
    users = client.get("/api/users").json()
    assert any(u["user_id"] == uid and u["group_name"] == "feynman" for u in users)


# ==================== E 阶段 Loop 工程化（PLAN 20） ====================


def _reset_user(user):
    """清空某用户的业务数据（保证测试幂等，防真实库污染）。"""
    import db
    db.get_user(user)
    with db._conn() as c:
        for t in ("assessments", "exercise_logs", "knowledge_points", "profile",
                  "reflow_logs"):
            c.execute(f"DELETE FROM {t} WHERE user_id=?", (user,))


def test_posttest_triggers_reflow_and_retest_completes():
    """E1：后测不达标 → 生成回流；重测达标 → 闭环完成。全走 API。"""
    import db
    import learning_pack

    user = "u_api_loop"
    _reset_user(user)
    pre = learning_pack.load_pretest()
    post = learning_pack.load_posttest()
    right_pre = {q["ex_id"]: str(q["check"]["answer"]) for q in pre}
    right_post = {q["ex_id"]: str(q["check"]["answer"]) for q in post}
    wrong = {q["ex_id"]: "999" for q in post}

    # 前测全对（score=1.0）→ 后测全错（0.0）→ 提升 -100pp，触发回流
    client.post("/api/quiz/python/pretest/submit",
                json={"user_id": user, "answers": right_pre, "mode": "feynman"})
    r = client.post("/api/quiz/python/posttest/submit",
                    json={"user_id": user, "answers": wrong, "mode": "feynman"})
    assert r.status_code == 200
    rf = r.json()["reflow"]
    assert rf["triggered"] is True
    assert rf["round"] == 1
    assert rf["weak_kps"]  # 后测答错的知识点即回流任务

    # 回流状态：active，有薄弱点
    s = client.get(f"/api/reflow/python/{user}").json()
    assert s["active"] is True
    assert s["round"] == 1
    assert s["weak_kps"]

    # 重测全对（达标）→ 闭环完成
    r2 = client.post("/api/quiz/python/posttest/submit",
                     json={"user_id": user, "answers": right_post, "mode": "feynman"})
    rf2 = r2.json()["reflow"]
    assert rf2["passed"] is True
    s2 = client.get(f"/api/reflow/python/{user}").json()
    assert s2["active"] is False
    assert s2["last_status"] == "completed"


def test_grade_returns_strategy_escalation():
    """E2：同一 kp 连续失败 → /api/grade 返回降级策略（explain/prereq）。"""
    import db
    import learning_pack

    user = "u_api_strat"
    _reset_user(user)
    # 选一个有前置知识点的 kp 的 mcq 题（prereq 策略才有前置可建议）
    graph = learning_pack.load_graph()
    kp_with_prereq = next(
        kp["kp_id"] for kp in graph["knowledge_points"] if kp.get("prerequisites"))
    ex = next(e for e in learning_pack.load_exercises()
              if e["type"] == "mcq" and e.get("kp_id") == kp_with_prereq)
    wrong = "999"
    r1 = client.post("/api/grade",
                     json={"user_id": user, "ex_id": ex["ex_id"], "answer": wrong})
    assert r1.json()["strategy"] == "hint"
    r2 = client.post("/api/grade",
                     json={"user_id": user, "ex_id": ex["ex_id"], "answer": wrong})
    assert r2.json()["strategy"] == "explain"
    r3 = client.post("/api/grade",
                     json={"user_id": user, "ex_id": ex["ex_id"], "answer": wrong})
    assert r3.json()["strategy"] == "prereq"
    assert r3.json().get("prereq_titles")  # 前置知识点建议


def test_queue_endpoint_returns_weak_kps():
    """E3：答错过的知识点进今日队列（按掌握度排序）。"""
    import db
    import learning_pack

    user = "u_api_queue"
    _reset_user(user)
    ex = next(e for e in learning_pack.load_exercises() if e["type"] == "mcq")
    client.post("/api/grade",
                json={"user_id": user, "ex_id": ex["ex_id"], "answer": "999"})
    d = client.get(f"/api/queue/python/{user}").json()
    q = d["queue"]
    assert any(item["kp_id"] == ex["kp_id"] and item["reason"] == "weak" for item in q)
