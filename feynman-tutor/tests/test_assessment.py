"""assessor/grader/db 集成测试（不依赖 LLM 的部分）。

评估 Agent 的前后测配对比对：pretest/posttest 必须同知识点、不同题。
"""

import db
import learning_pack


def test_pretest_posttest_pairs_same_kp():
    """前后测知识点一一对应，且题目确实不同（P0 验收硬约束）。"""
    pairs = learning_pack.pretest_posttest_pairs()
    assert len(pairs) == 10
    for pre, post in pairs:
        assert pre["kp_id"] == post["kp_id"], "前后测知识点必须对应"
        assert pre["ex_id"] != post["ex_id"], "前后测必须是不同题（防记住答案）"
        assert pre["type"] == post["type"] == "mcq"
        assert pre["difficulty"] == post["difficulty"], "同难度"


def test_exercises_all_rule_gradable():
    """40 题全部可规则判：type 必须是 output/code/mcq 之一，且有对应 check。"""
    exs = learning_pack.load_exercises()
    assert len(exs) == 40
    for ex in exs:
        t = ex["type"]
        assert t in ("output", "code", "mcq")
        if t == "output":
            assert "expect_stdout" in ex["check"]
        elif t == "code":
            assert "tests" in ex["check"]
        elif t == "mcq":
            assert "options" in ex["check"] and "answer" in ex["check"]
            assert 0 <= ex["check"]["answer"] < len(ex["check"]["options"])


def test_exercises_kp_coverage():
    """12 个知识点每个都有练习题。"""
    graph = learning_pack.load_graph()
    by_kp = learning_pack.exercises_by_kp()
    for kp in graph["knowledge_points"]:
        assert kp["kp_id"] in by_kp, f"{kp['kp_id']} 缺练习题"


def test_db_multi_user_isolation(tmp_path):
    """多用户数据隔离：u0 的答题记录不影响 u1。"""
    p = str(tmp_path / "t.db")
    db.init_db(p)
    db.get_user("u0", db_path=p)
    db.get_user("u1", db_path=p)
    db.log_exercise("u0", "ex1", "kp1", True, "ans", "ok", db_path=p)
    assert len(db.get_exercise_logs("u0", db_path=p)) == 1
    assert len(db.get_exercise_logs("u1", db_path=p)) == 0
    # 知识点同样隔离
    db.upsert_kp("u0", {"kp_id": "k", "title": "t", "chapter": "c"}, db_path=p)
    assert db.get_kp("u0", "k", db_path=p) is not None
    assert db.get_kp("u1", "k", db_path=p) is None
