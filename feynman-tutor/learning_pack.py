"""学习包加载器：读 learning_packs/<course>/ 下的结构化数据。

对外只暴露几个纯函数，agent 层不直接碰文件格式。
"""

import json
from pathlib import Path

PACKS_DIR = Path(__file__).resolve().parent / "learning_packs"


class LearningPackError(Exception):
    """学习包缺失/格式错误。"""


def load_graph(course: str = "python") -> dict:
    """知识图谱：{kp_id: {...}} 索引 + chapters 元信息。"""
    p = PACKS_DIR / course / "knowledge-graph.json"
    if not p.exists():
        raise LearningPackError(f"学习包不存在: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["_by_id"] = {kp["kp_id"]: kp for kp in raw["knowledge_points"]}
    return raw


def load_exercises(course: str = "python") -> list[dict]:
    return _load_jsonl(PACKS_DIR / course / "exercises.jsonl")


def load_pretest(course: str = "python") -> list[dict]:
    return _load_jsonl(PACKS_DIR / course / "pretest.jsonl")


def load_posttest(course: str = "python") -> list[dict]:
    return _load_jsonl(PACKS_DIR / course / "posttest.jsonl")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise LearningPackError(f"题目文件不存在: {path}")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def exercises_by_kp(course: str = "python") -> dict[str, list[dict]]:
    """按知识点分组练习题，供练习环节按 kp 取题。"""
    grouped: dict[str, list[dict]] = {}
    for ex in load_exercises(course):
        grouped.setdefault(ex["kp_id"], []).append(ex)
    return grouped


def notes_for(course: str, kp_id: str) -> str:
    """知识点的精编兜底讲解（md 文本）。"""
    graph = load_graph(course)
    kp = graph["_by_id"].get(kp_id)
    if not kp:
        return ""
    p = PACKS_DIR / course / kp.get("notes_file", f"notes/{kp_id}.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""


def pretest_posttest_pairs(course: str = "python") -> list[tuple[dict, dict]]:
    """前测/后测按知识点配对（同 kp 一一对应）。"""
    pre = load_pretest(course)
    post = load_posttest(course)
    post_by_kp = {q["kp_id"]: q for q in post}
    pairs = []
    for p in pre:
        q = post_by_kp.get(p["kp_id"])
        if q:
            pairs.append((p, q))
    return pairs
