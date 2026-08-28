"""推荐 Agent（P1）：按画像薄弱点 + 答题记录选练习。

规则优先（防漂移，PLAN 10.3）：选题排序纯规则决定——
1. 薄弱点且从未做过/做错的题优先；
2. 其次薄弱点已做对但需要巩固的；
3. 最后非薄弱点新题。
LLM 只为每道推荐题生成 1 句"为什么推荐"（失败退回规则文案）。

输出：{"recommendations": [{"ex_id", "kp_id", "reason"}], "by_rule": True}
"""

import json

import db
import learning_pack
import model
import prompts

# 推荐 prompt 已版本化：prompts/recommender.md（D3，PLAN 17.2）


def _load_attempt_map(user_id: str, db_path: str | None) -> dict[str, dict]:
    """答题记录 → {ex_id: {correct: bool, attempts: int}}（取最近一次为准）。"""
    logs = db.get_exercise_logs(user_id, db_path=db_path)
    m: dict[str, dict] = {}
    for log in logs:
        ex_id = log.get("ex_id")
        if not ex_id:
            continue
        if ex_id not in m or log.get("id", 0) > m[ex_id].get("_id", 0):
            m[ex_id] = {"correct": bool(log.get("correct")), "attempts": 1,
                        "_id": log.get("id", 0)}
    for v in m.values():
        v.pop("_id", None)
    return m


def recommend(user_id: str, course: str = "python", top_n: int = 5,
              db_path: str | None = None) -> dict:
    """按薄弱点+错题推荐练习题。

    返回 {"recommendations": [{"ex_id","kp_id","reason"}], "by_rule": True}
    """
    profile = db.get_profile(user_id, db_path=db_path) or {}
    weak = db.parse_weak_ids(profile)
    if not weak:
        # 无画像：推荐全量题目里从未做过的（按知识点顺序）
        weak = [kp["kp_id"] for kp in learning_pack.load_graph(course)["knowledge_points"]]

    attempts = _load_attempt_map(user_id, db_path)
    graph = learning_pack.load_graph(course)
    titles = {kp["kp_id"]: kp["title"] for kp in graph["knowledge_points"]}

    # 掌握门槛：blocked 的 kp 及其依赖者不推荐（先回去巩固基础）
    by_id = graph["_by_id"]

    def dependents(kid: str) -> set[str]:
        result = {kid}
        for other, kp in by_id.items():
            if kid in kp.get("prerequisites", []) and other in by_id:
                result |= dependents(other)
        return result

    kps = db.list_kps(user_id, db_path=db_path)
    blocked_set: set[str] = set()
    for k in kps:
        if k.get("status") == "blocked" and k["kp_id"] in by_id:
            blocked_set |= dependents(k["kp_id"])
    weak = [w for w in weak if w not in blocked_set]

    # 规则排序
    candidates: list[tuple[int, int, dict]] = []  # (priority, order, ex)
    for kp_id in weak:
        for ex in learning_pack.exercises_by_kp(course).get(kp_id, []):
            st = attempts.get(ex["ex_id"])
            if st is None:
                priority = 0  # 薄弱点且未做过：最优先
            elif not st["correct"]:
                priority = 1  # 薄弱点做错过
            else:
                priority = 2  # 薄弱点已做对（巩固）
            candidates.append((priority, 0, ex))
    # 非薄弱点新题补位
    done = {ex["ex_id"] for _, _, ex in candidates}
    for kp_id, kps in learning_pack.exercises_by_kp(course).items():
        if kp_id in weak or kp_id in blocked_set:
            continue
        for ex in kps:
            if ex["ex_id"] not in done and ex["ex_id"] not in attempts:
                candidates.append((3, 1, ex))

    candidates.sort(key=lambda x: (x[0], x[1]))
    picked = [ex for _, _, ex in candidates[:top_n]]

    # LLM 生成理由（失败退回规则文案）
    reasons = _reasons(weak, titles, picked)
    recs = [{"ex_id": ex["ex_id"], "kp_id": ex.get("kp_id", ""),
             "reason": reasons.get(ex["ex_id"],
                                   f"针对薄弱点 {titles.get(ex.get('kp_id',''), ex.get('kp_id',''))} 的练习")}
            for ex in picked]
    return {"recommendations": recs, "by_rule": True}


def _reasons(weak: list[str], titles: dict[str, str], picked: list[dict]) -> dict:
    """LLM 批量生成推荐理由。失败返回空 dict（调用方用规则文案兜底）。"""
    if not picked:
        return {}
    items = "\n".join(f"  {ex['ex_id']}: {ex.get('kp_id','')}" for ex in picked)
    try:
        raw = model.chat(
            [{"role": "system", "content": "你只输出合法 JSON。"},
             {"role": "user", "content": prompts.load("recommender.md").format(
                 weak=", ".join(titles.get(w, w) for w in weak[:5]),
                 titles=json.dumps(titles, ensure_ascii=False),
                 items=items)}],
            temperature=0.2, max_tokens=800, caller="recommender",
        )
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return {str(k): str(v) for k, v in data.get("reasons", {}).items()}
    except (json.JSONDecodeError, ValueError, model.ModelError):
        return {}
