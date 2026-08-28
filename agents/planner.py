"""规划 Agent（P1）：画像 × 知识图谱 → 定制学习路径。

思路（PLAN 3.3 的"规划 Agent"职责）：
1. 从知识图谱构建前置依赖 DAG，拓扑排序得"必须遵守的基础顺序"（前置先学）；
2. 叠加用户画像：薄弱点优先排前（在依赖约束内）、已掌握/高掌握度靠后或标注"可跳过"；
3. LLM 只负责把路径转成可读的"为什么这么排"（防漂移：路径本身是规则算的，LLM 不改路径）。

输出：{"path": [kp_id...], "rationale": 可读说明, "by_rule": True}
LLM 失败时 rationale 退回规则模板文案，路径始终由规则产生。
"""

import json
from collections import deque

import db
import learning_pack
import model
import prompts

# 规划 prompt 已版本化：prompts/planner.md（D3，PLAN 17.2）


def _topo_sort(graph: dict) -> list[str]:
    """Kahn 拓扑排序：前置知识点必须先学。返回 kp_id 序列（依赖顺序）。"""
    by_id = graph["_by_id"]
    indegree = {kid: len(kp.get("prerequisites", [])) for kid, kp in by_id.items()}
    children: dict[str, list[str]] = {kid: [] for kid in by_id}
    for kid, kp in by_id.items():
        for pre in kp.get("prerequisites", []):
            if pre in children:
                children[pre].append(kid)
    # 不存在的前置（防学习包数据错误）从 indegree 里扣掉
    for kid, kp in by_id.items():
        for pre in kp.get("prerequisites", []):
            if pre not in by_id:
                indegree[kid] -= 1

    queue = deque(sorted(k for k, d in indegree.items() if d == 0))
    order: list[str] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in sorted(children[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(by_id):  # 有环或孤点：兜底直接按原顺序
        return [kp["kp_id"] for kp in graph["knowledge_points"]]
    return order


def _reorder_by_profile(graph: dict, order: list[str],
                        weak: list[str], mastery: dict[str, float],
                        status: dict[str, str] | None = None) -> list[str]:
    """在依赖约束内把薄弱点提前，把卡住（blocked）的及其依赖者暂缓。

    关键约束：薄弱点 k 被提前时，k 的所有前置（递归）也必须跟着提前，
    否则会破坏"前置先学"（如 string.methods 依赖 string.basic，前者被提前后
    basic 还排在后面就是错的）。实现：对薄弱点集合做"前置闭包"，
    闭包整体按原拓扑序提前，其余保持原序；已掌握（mastery>=0.8）排最后。
    掌握门槛（Mastery gate）：练习 3 次未通过标 blocked，其本身 + 依赖它的
    知识点（递归依赖闭包）暂缓到最后——未掌握的基石不硬推后续内容。
    """
    by_id = graph["_by_id"]

    def closure(kid: str) -> set[str]:
        """递归收集前置闭包。"""
        result = {kid}
        for pre in by_id[kid].get("prerequisites", []):
            if pre in by_id:
                result |= closure(pre)
        return result

    def dependents(kid: str) -> set[str]:
        """递归收集依赖者闭包（哪些知识点以 kid 为前置）。"""
        result = {kid}
        for other, kp in by_id.items():
            if kid in kp.get("prerequisites", []) and other in by_id:
                result |= dependents(other)
        return result

    status = status or {}
    weak_set = set(weak)
    # 薄弱闭包：每个薄弱点的前置都拉进来（保持拓扑约束）
    promote_set: set[str] = set()
    for k in weak_set:
        promote_set |= closure(k)

    # 掌握门槛：blocked 的 kp 及其递归依赖者全部暂缓
    blocked_set: set[str] = set()
    for kid, st in status.items():
        if st == "blocked" and kid in by_id:
            blocked_set |= dependents(kid)

    # 按原拓扑序切组：薄弱闭包 / 其余 / 已掌握 / 卡住（blocked 及其依赖）
    rest = [k for k in order if k not in promote_set and k not in blocked_set]
    mastered = [k for k in rest if mastery.get(k, 0) >= 0.8]
    normal = [k for k in rest if mastery.get(k, 0) < 0.8]
    promoted = [k for k in order if k in promote_set]  # 保持内部拓扑序
    blocked = [k for k in order if k in blocked_set]   # 保持内部拓扑序，排最后
    return promoted + normal + mastered + blocked


def plan_path(user_id: str, course: str = "python",
              db_path: str | None = None) -> dict:
    """生成用户学习路径。

    返回 {"path": [kp_id...], "titles": {kp_id: title}, "rationale": str, "by_rule": True}
    """
    graph = learning_pack.load_graph(course)
    order = _topo_sort(graph)

    # 画像数据（无画像则全量返回拓扑序）
    profile = db.get_profile(user_id, db_path=db_path) or {}
    weak = db.parse_weak_ids(profile)
    kps = db.list_kps(user_id, db_path=db_path)
    mastery = {k["kp_id"]: k.get("mastery", 0) or 0 for k in kps}
    status = {k["kp_id"]: k.get("status") or "new" for k in kps}

    path = _reorder_by_profile(graph, order, weak, mastery, status)
    titles = {kp["kp_id"]: kp["title"] for kp in graph["knowledge_points"]}

    # LLM 生成可读解释；失败退回规则模板（路径不改）
    rationale = _rationale(weak, profile, path, titles, status)
    return {"path": path, "titles": titles, "rationale": rationale, "by_rule": True}


def _rationale(weak: list[str], profile: dict, path: list[str],
               titles: dict[str, str], status: dict[str, str] | None = None) -> str:
    """生成路径解释。LLM 失败/无画像时用规则模板。"""
    title_map = "\n".join(f"  {k}: {titles[k]}" for k in path)
    status = status or {}
    blocked = [k for k, st in status.items() if st == "blocked"]
    if not weak and not blocked:
        return ("你还没有足够的答题记录，先按基础顺序学：前置知识点优先。"
                "学完前测+练习后我会重新规划。")
    try:
        raw = model.chat(
            [{"role": "system", "content": "你是学习规划专家，直接输出解释文本，不要 JSON。"},
             {"role": "user", "content": prompts.load("planner.md").format(
                 weak=", ".join(titles.get(w, w) for w in weak) or "暂无",
                 avg_correct=profile.get("avg_correct", 0),
                 style=profile.get("learning_style", "简答"),
                 path=" → ".join(f"{k}({titles.get(k, k)})" for k in path),
                 title_map=title_map)}],
            temperature=0.3, max_tokens=400, caller="planner",
        )
        return raw.strip() or "路径已按依赖和薄弱点规划，见上方顺序。"
    except model.ModelError:
        lines = ["路径按规则规划：先满足前置依赖。"]
        if weak:
            weak_names = ", ".join(titles.get(w, w) for w in weak[:3])
            lines.append(f"你较薄弱的 {weak_names} 已提前，建议优先学习。")
        if blocked:
            blocked_names = ", ".join(titles.get(k, k) for k in blocked[:3])
            lines.append(f"练习连续未通过的 {blocked_names} 已暂缓，先回去巩固基础。")
        lines.append("已掌握的排最后，可快速跳过。")
        return " ".join(lines)
