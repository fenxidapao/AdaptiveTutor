"""6 步闭环全流程冒烟（D2 降级方案：TestClient + mock LLM，进 CI 常跑）。

价值：真实浏览器 E2E（scripts/e2e_flow.py，需 playwright chromium）覆盖渲染层，
本测试覆盖全部路由与判题链路的契约——注册→前测→诊断→费曼→练习→后测→报告。
"""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    """打桩 LLM：按 caller 返回固定格式（快、免费、可重复）。"""
    def fake(messages, temperature=0.3, max_tokens=2000, caller=None, **kw):
        c = caller or ""
        if "diagnostic" in c:
            return ('{"weak_points":[{"kp_id":"python.list.slice","reason":"做错切片题",'
                    '"evidence":["py.ch1.d.1"]}],"learning_style":"简答","avg_correct":0.5}')
        if "gaps" in c:
            return '{"gaps":["切片 end 不含在结果里"]}'
        if "rag" in c:
            return '{"used":[0],"explanation":"列表切片用冒号取子集，start 含 end 不含。"}'
        if "recommender" in c:
            return '{"reasons":{"py.ch1.d.1":"补切片薄弱点"}}'
        if "planner" in c:
            return "按前置依赖先学基础，再优先攻克切片。"
        if "hint" in c:
            return "想想切片 end 是否含在结果里。"
        return "很好，再想想边界情况？"
    monkeypatch.setattr(model, "chat", fake)
    monkeypatch.setattr(model, "chat_with_fallback", fake)


def test_full_6step_flow():
    """注册 → 前测 → 诊断 → 费曼 3 轮 → 练习 → 后测 → 报告，全链路契约。"""
    uid = f"u_flow_{uuid.uuid4().hex[:6]}"

    # 1 注册
    r = client.post("/api/register", json={"user_id": uid, "password": "pass123"})
    assert r.status_code == 200
    sid = r.json()["session_id"]

    def qs(p):
        return {**p, "user_id": uid, "session_id": sid}

    # 2 前测（全对 → 1.0）
    pre = client.get(f"/api/quiz/python/pretest?user_id={uid}&session_id={sid}").json()
    ans = {}
    for q in pre:
        import learning_pack
        full = learning_pack.load_pretest("python")
        ex = next(x for x in full if x["ex_id"] == q["ex_id"])
        ans[q["ex_id"]] = str(ex["check"]["answer"]) if ex["type"] == "mcq" else "print(1)"
    r = client.post("/api/quiz/python/pretest/submit", json=qs({
        "answers": ans, "elapsed_seconds": 60}))
    assert r.status_code == 200 and r.json()["score"] == 1.0

    # 3 诊断（mock LLM → 画像）
    r = client.post(f"/api/diagnose/{uid}?session_id={sid}")
    assert r.status_code == 200
    assert r.json()["weak_points"][0]["kp_id"] == "python.list.slice"

    # 4 费曼 3 轮 → 盲点 → 讲解 → 练习判题
    for _ in range(3):
        r = client.post("/api/feynman/turn", json=qs({
            "course": "python", "kp_id": "python.list.slice",
            "transcript": [{"role": "user", "content": "切片就是 a[1:3] 取一段"}]}))
        assert r.status_code == 200 and r.json()["coach"]
    r = client.post("/api/feynman/summarize", json=qs({
        "course": "python", "kp_id": "python.list.slice",
        "transcript": [{"role": "user", "content": "切片 a[1:3]"}]}))
    assert r.status_code == 200 and r.json()["gaps"]
    r = client.get(f"/api/explain/python/python.list.slice?user_id={uid}&session_id={sid}")
    assert r.status_code == 200 and "列表切片" in r.json()["explanation"]

    # 练习：取一道题判题（答错 → 有考点/反馈；mock hint 只给方向）
    exs = client.get(f"/api/exercises/python?kp_id=python.list.slice").json()
    assert exs
    r = client.post("/api/grade", json=qs({"ex_id": exs[0]["ex_id"], "answer": "1+1"}))
    assert r.status_code == 200
    assert "correct" in r.json()

    # 5 后测（mock 判题全对）
    post = client.get(f"/api/quiz/python/posttest?user_id={uid}&session_id={sid}").json()
    ans2 = {}
    for q in post:
        import learning_pack
        full = learning_pack.load_posttest("python")
        ex = next(x for x in full if x["ex_id"] == q["ex_id"])
        ans2[q["ex_id"]] = str(ex["check"]["answer"]) if ex["type"] == "mcq" else "print(1)"
    r = client.post("/api/quiz/python/posttest/submit", json=qs({
        "answers": ans2, "elapsed_seconds": 60}))
    assert r.status_code == 200 and r.json()["score"] == 1.0

    # 6 报告（数字与库一致）
    r = client.get(f"/api/report/python/{uid}?session_id={sid}")
    assert r.status_code == 200
    assert r.json()["pre"] == 1.0 and r.json()["post"] == 1.0
    assert "gain_pp" in r.json()

    # 热力图也通
    r = client.get(f"/api/heatmap/{uid}?session_id={sid}")
    assert r.status_code == 200 and len(r.json()["cells"]) == 12
