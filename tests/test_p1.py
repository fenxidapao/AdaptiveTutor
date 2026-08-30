"""P1 测试：规划 Agent（依赖约束/薄弱优先）、推荐 Agent（错题优先）、多用户分组。

全部不依赖真实 LLM：planner/recommender 的核心逻辑是纯规则，
LLM 只生成可读文本，测试只验证规则部分。
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


def _graph():
    return learning_pack.load_graph()


def test_topo_sort_respects_prerequisites():
    """拓扑排序：任何知识点的前置必须排在其前。"""
    from agents import planner
    graph = _graph()
    order = planner._topo_sort(graph)
    assert len(order) == 12
    idx = {k: i for i, k in enumerate(order)}
    for k in order:
        for pre in graph["_by_id"][k].get("prerequisites", []):
            assert idx[pre] < idx[k], f"{pre} 应排在 {k} 前"


def test_reorder_weak_promotes_with_closure():
    """薄弱点提前时，其前置闭包也跟着提前（不破坏依赖）。"""
    from agents import planner
    graph = _graph()
    order = planner._topo_sort(graph)
    # string.methods 依赖 string.basic → variable.assignment
    path = planner._reorder_by_profile(
        graph, order, ["python.string.methods"], {})
    idx = {k: i for i, k in enumerate(path)}
    # 薄弱点本身靠前，且前置都在它前面
    assert idx["python.string.methods"] < 5
    for pre in graph["_by_id"]["python.string.methods"]["prerequisites"]:
        assert idx[pre] < idx["python.string.methods"]
    # 校验全路径无依赖违规
    for k in path:
        for pre in graph["_by_id"][k].get("prerequisites", []):
            assert idx[pre] < idx[k]


def test_reorder_mastered_last():
    """已掌握（mastery>=0.8）的知识点排最后。"""
    from agents import planner
    graph = _graph()
    order = planner._topo_sort(graph)
    mastery = {"python.variable.assignment": 0.9,
               "python.int.float": 0.85}
    path = planner._reorder_by_profile(graph, order, [], mastery)
    assert path[-2:] == ["python.variable.assignment", "python.int.float"] or \
           path[-1] in ("python.variable.assignment", "python.int.float")


def test_plan_path_no_profile_returns_topo(tmp_db):
    """无画像用户：返回纯拓扑序，不报错。"""
    from agents import planner
    db.get_user("u_new", db_path=tmp_db)
    plan = planner.plan_path("u_new", db_path=tmp_db)
    assert len(plan["path"]) == 12
    assert plan["by_rule"] is True


def test_recommend_weak_untried_first(tmp_db):
    """推荐排序：薄弱点未做过的题排最前。"""
    from agents import recommender
    user = "u_rec"
    db.get_user(user, db_path=tmp_db)
    # 造画像：薄弱点 = list.slice
    import json
    db.save_profile(user, {"weak_points": ["python.list.slice"],
                           "learning_style": "代码", "avg_correct": 0.4},
                    db_path=tmp_db)
    r = recommender.recommend(user, top_n=3, db_path=tmp_db)
    assert r["recommendations"]
    # 第一条必须是 list.slice 的题
    assert r["recommendations"][0]["kp_id"] == "python.list.slice"


def test_recommend_wrong_first_then_untried(tmp_db):
    """同一知识点：做错/未做的题排在已做对的前面（priority: 未做0 < 错1 < 对2）。"""
    from agents import recommender
    user = "u_wrong"
    db.get_user(user, db_path=tmp_db)
    import json
    db.save_profile(user, {"weak_points": ["python.list.basic"],
                           "learning_style": "代码", "avg_correct": 0.5},
                    db_path=tmp_db)
    # py.ch2.l.1 做错过，py.ch2.l.2 做对过，py.ch2.l.3 未做
    db.log_exercise(user, "py.ch2.l.1", "python.list.basic", False,
                    "a[0]", "错", db_path=tmp_db)
    db.log_exercise(user, "py.ch2.l.2", "python.list.basic", True,
                    "a[1]", "对", db_path=tmp_db)
    r = recommender.recommend(user, top_n=5, db_path=tmp_db)
    ex_order = [rec["ex_id"] for rec in r["recommendations"]
                if rec["kp_id"] == "python.list.basic"]
    # 未做的 l.3 和做错的 l.1 都应在做对的 l.2 前面
    assert ex_order.index("py.ch2.l.3") < ex_order.index("py.ch2.l.2")
    assert ex_order.index("py.ch2.l.1") < ex_order.index("py.ch2.l.2")


def test_user_register_and_group(tmp_db):
    """注册带 name + 分组 + 列表。"""
    u = db.get_user("u_test2", name="测试同学", db_path=tmp_db)
    assert u["name"] == "测试同学"
    assert u["group_name"] is None
    u2 = db.assign_group("u_test2", "feynman", db_path=tmp_db)
    assert u2["group_name"] == "feynman"
    users = db.list_users(db_path=tmp_db)
    assert any(x["user_id"] == "u_test2" and x["group_name"] == "feynman"
               for x in users)


def test_mastery_gate_blocked_status(tmp_db):
    """掌握门槛：同一 kp 连续 3 次全错 → status=blocked。"""
    user = "u_blocked"
    db.get_user(user, db_path=tmp_db)
    for i in range(3):
        db.log_exercise(user, f"e{i}", "python.list.slice", False,
                        "x", "错", db_path=tmp_db)
    kp = db.get_kp(user, "python.list.slice", db_path=tmp_db)
    assert kp["status"] == "blocked"


def test_mastery_gate_unblocks_after_correct(tmp_db):
    """答对一次后 blocked 解除（回到 learning/mastered）。"""
    user = "u_unblock"
    db.get_user(user, db_path=tmp_db)
    for i in range(3):
        db.log_exercise(user, f"e{i}", "python.list.slice", False,
                        "x", "错", db_path=tmp_db)
    assert db.get_kp(user, "python.list.slice", db_path=tmp_db)["status"] == "blocked"
    db.log_exercise(user, "e3", "python.list.slice", True,
                    "a[1:3]", "对", db_path=tmp_db)
    kp = db.get_kp(user, "python.list.slice", db_path=tmp_db)
    assert kp["status"] != "blocked"
    assert kp["mastery"] > 0


def test_planner_defers_blocked_dependents(tmp_db):
    """掌握门槛：blocked 的 kp 及其依赖者排到路径末尾。"""
    from agents import planner
    user = "u_plan_blocked"
    db.get_user(user, db_path=tmp_db)
    # string.methods 是基础且被 list 等依赖；让它 blocked
    for i in range(3):
        db.log_exercise(user, f"e{i}", "python.string.methods", False,
                        "x", "错", db_path=tmp_db)
    plan = planner.plan_path(user, db_path=tmp_db)
    idx = {k: i for i, k in enumerate(plan["path"])}
    # string.methods 本身靠后，且依赖它的 kp 也在其后
    assert idx["python.string.methods"] > 6
    # 依赖 string.methods 的（如 string.basic 之后的内容）不应在它前面
    for k in plan["path"]:
        for pre in learning_pack.load_graph()["_by_id"][k].get("prerequisites", []):
            assert idx[pre] < idx[k]


def test_recommender_skips_blocked(tmp_db):
    """掌握门槛：blocked 的 kp 不进入推荐。"""
    from agents import recommender
    user = "u_rec_blocked"
    db.get_user(user, db_path=tmp_db)
    for i in range(3):
        db.log_exercise(user, f"e{i}", "python.string.methods", False,
                        "x", "错", db_path=tmp_db)
    db.save_profile(user, {"weak_points": ["python.string.methods"],
                           "learning_style": "代码", "avg_correct": 0.3},
                    db_path=tmp_db)
    r = recommender.recommend(user, top_n=10, db_path=tmp_db)
    kps = {rec["kp_id"] for rec in r["recommendations"]}
    assert "python.string.methods" not in kps


def test_recommend_fallback_when_all_blocked(tmp_db, monkeypatch):
    """复核队列修复回归：全错新手被 blocked+依赖闭包排除到无题可推时，
    兜底推荐 blocked 知识点自身的题（做错的优先重练），推荐列表不得为空。"""
    import model
    from agents import recommender

    def _offline(*a, **k):
        raise model.ModelError("测试离线")

    monkeypatch.setattr(model, "chat", _offline)
    user = "u_rec_allblocked"
    db.get_user(user, db_path=tmp_db)
    graph = learning_pack.load_graph()
    by_kp = learning_pack.exercises_by_kp()
    # 造"每个 kp 连错 3 次（真实题目 id）"的极端全错新手 → 全部 blocked，
    # 依赖闭包排除整图，主路径无题可推
    wrong_ids: set[str] = set()
    for kp in graph["knowledge_points"]:
        for ex in by_kp.get(kp["kp_id"], [])[:3]:
            db.log_exercise(user, ex["ex_id"], kp["kp_id"], False,
                            "x", "错", db_path=tmp_db)
            wrong_ids.add(ex["ex_id"])
    db.save_profile(user, {
        "weak_points": [kp["kp_id"] for kp in graph["knowledge_points"]],
        "learning_style": "代码", "avg_correct": 0.0}, db_path=tmp_db)
    r = recommender.recommend(user, top_n=3, db_path=tmp_db)
    assert r["recommendations"], "全错新手也必须拿到推荐（兜底）"
    assert len(r["recommendations"]) == 3
    # 兜底优先给做错过的题重练
    assert all(rec["ex_id"] in wrong_ids for rec in r["recommendations"])
