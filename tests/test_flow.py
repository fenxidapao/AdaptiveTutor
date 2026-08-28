"""闭环集成测试（不依赖真实 LLM）：验证流程控制与状态写入。

策略：mock model.chat 返回固定文本 + 预置输入队列，覆盖
前测 → 诊断 → 费曼 → 练习 → 后测 → 报告 的完整状态流转。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
import learning_pack  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    p = str(tmp_path / "state.db")
    db.init_db(p)
    return p


def _mcq_answers():
    ans = {}
    for q in learning_pack.load_pretest() + learning_pack.load_posttest():
        if q["type"] == "mcq":
            ans[q["ex_id"]] = str(q["check"]["answer"])
    for ex in learning_pack.load_exercises():
        if ex["type"] == "mcq":
            ans[ex["ex_id"]] = str(ex["check"]["answer"])
    return ans


def test_pretest_writes_state(tmp_db, monkeypatch):
    """前测：答案全部入 exercise_logs + assessments。"""
    from agents import assessor

    user = "u_test"
    db.get_user(user, db_path=tmp_db)
    ans = _mcq_answers()
    pre = learning_pack.load_pretest()
    q = iter(ans[q["ex_id"]] for q in pre)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(q))

    rate = assessor.run_pretest(user, None, "feynman", "python",
                                ask_user=lambda p: next(q), db_path=tmp_db)
    assert 0.0 <= rate <= 1.0
    logs = db.get_exercise_logs(user, db_path=tmp_db)
    assert len(logs) == len(pre)  # 每前测题一条日志
    ass = db.get_assessments(user, None, tmp_db)
    assert any(a["kind"] == "pretest" for a in ass)


def test_diagnose_fallback_on_no_data(tmp_db):
    """无答题记录时诊断返回空画像（不抛错）。"""
    from agents import diagnostic
    p = diagnostic.diagnose("u_new", db_path=tmp_db)
    assert p["weak_points"] == []
    assert p["avg_correct"] == 0.0


def test_diagnose_stats_fallback(tmp_db, monkeypatch):
    """LLM 失败时统计兜底：薄弱点按正确率取最低。"""
    from agents import diagnostic
    import model

    user = "u_stats"
    db.get_user(user, db_path=tmp_db)
    db.log_exercise(user, "e1", "kp.a", False, "x", "wrong", db_path=tmp_db)
    db.log_exercise(user, "e2", "kp.a", False, "x", "wrong", db_path=tmp_db)
    db.log_exercise(user, "e3", "kp.b", True, "x", "ok", db_path=tmp_db)

    def boom(*a, **k):
        raise model.ModelError("模拟 API 故障")

    monkeypatch.setattr(model, "chat", boom)
    p = diagnostic.diagnose(user, db_path=tmp_db)
    assert p["fallback"] is True
    weak_ids = [w["kp_id"] if isinstance(w, dict) else w for w in p["weak_points"]]
    assert "kp.a" in weak_ids  # 全错的知识点必在薄弱点
    assert p["avg_correct"] == pytest.approx(1 / 3, abs=0.01)


def test_explain_kp_notes_fallback(tmp_db, monkeypatch):
    """RAG 检索失败 → notes 兜底（不抛错）。"""
    from agents import feynman
    import rag

    user = "u_notes"
    db.get_user(user, db_path=tmp_db)

    def boom(*a, **k):
        raise rag.RAGError("CourseRAG 不可用")

    monkeypatch.setattr(rag, "_retrieve_raw", boom)
    text = feynman.explain_kp(user, "python.list.slice", db_path=tmp_db)
    assert "切片" in text  # notes 兜底内容
    kp = db.get_kp(user, "python.list.slice", db_path=tmp_db)
    assert kp["explain_count"] >= 1


def test_hint_only_no_llm_returns_default(tmp_db, monkeypatch):
    """LLM 故障时提示给默认文案（不抛错）。"""
    from agents import feynman
    import model

    def boom(*a, **k):
        raise model.ModelError("故障")

    monkeypatch.setattr(model, "chat", boom)
    ex = {"prompt": "题目", "check": {}}
    hint = feynman.hint_only(ex, {"user_answer": "错", "feedback": "不对"})
    assert isinstance(hint, str) and len(hint) > 0


def test_learn_full_flow(monkeypatch, tmp_db, capsys):
    """完整闭环（mock LLM）：前测→诊断→费曼→练习→后测→报告，状态全部落库。

    mock model.chat 返回固定 JSON 诊断结果（薄弱点=ch1 前两个知识点），
    这样知识点顺序可预测，输入队列可精确构造。
    """
    import json
    import model
    from agents import assessor, diagnostic, feynman
    import main as main_mod

    user = "u_flow"
    # _learn_flow 内部用默认 config.DB_PATH，monkeypatch 指向临时库
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_db)
    db.init_db(tmp_db)
    db.get_user(user, db_path=tmp_db)
    db_path = tmp_db

    # ch1 全部知识点（_learn_flow 会对每个知识点过费曼+练习）
    ch1_kps = [kp["kp_id"] for kp in learning_pack.load_graph()["knowledge_points"]
               if kp["chapter"] == "ch1"]

    # --- mock LLM ---
    def fake_chat(messages, **kw):
        # 诊断 prompt 返回固定画像；其他返回讲解文本
        joined = messages[-1]["content"] if messages else ""
        if "weak_points" in joined:
            return json.dumps({"weak_points": ["python.variable.assignment",
                                               "python.int.float"],
                               "learning_style": "代码", "avg_correct": 0.6},
                              ensure_ascii=False)
        if "gaps" in joined:
            return json.dumps({"gaps": ["索引从0开始"]}, ensure_ascii=False)
        if "direction" in joined or "助教" in joined:
            return "想想索引从几开始？"
        return "变量赋值：用 = 给名字绑定值，如 x = 5。"

    monkeypatch.setattr(model, "chat", fake_chat)
    monkeypatch.setattr(model, "chat_with_fallback", fake_chat)

    # --- 构造输入队列 ---
    ans = _mcq_answers()
    pre = [q for q in learning_pack.load_pretest() if q.get("chapter") == "ch1"]
    post = [q for q in learning_pack.load_posttest() if q.get("chapter") == "ch1"]

    feed = [ans[q["ex_id"]] for q in pre]          # 前测
    from tests.verify_learning_pack import SAMPLES
    ex_group = learning_pack.exercises_by_kp()
    for kp in ch1_kps:                              # 每 kp：费曼 3 轮 → 练习 3 题（_learn_flow 交替）
        feed += ["变量就是名字", "能举例吗", "会了"]
        for ex in ex_group.get(kp, [])[:3]:
            if ex["type"] == "mcq":
                feed.append(ans.get(ex["ex_id"], "0"))
            else:
                feed.append(SAMPLES.get(ex["ex_id"], "print(1)"))
    feed += [ans[q["ex_id"]] for q in post]         # 后测

    it = iter(feed)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))

    # --- 直接调 main 内部流程（绕过 argparse） ---
    main_mod._learn_flow(user, "python", "feynman", "ch1")

    # --- 断言 ---
    ass = db.get_assessments(user, "ch1", tmp_db)
    kinds = {a["kind"] for a in ass}
    assert "pretest" in kinds and "posttest" in kinds
    logs = db.get_exercise_logs(user, db_path=tmp_db)
    assert len(logs) >= len(pre) + len(post)  # 前后测日志
    prof = db.get_profile(user, db_path=tmp_db)
    assert prof is not None
    # 后测记录正确率
    post_row = [a for a in ass if a["kind"] == "posttest"][-1]
    assert 0.0 <= post_row["score"] <= 1.0
