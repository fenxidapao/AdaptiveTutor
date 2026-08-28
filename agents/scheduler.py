"""间隔复习调度器（P2）：SM-2 简化算法，纯规则不调 LLM（防漂移）。

设计（抄 anki-mcp-server 思路，PLAN 4/REFERENCE 1.4）：
- 每个知识点维护 next_review（复习日期）+ review_interval（当前间隔档位，0=首次）；
- 复习时取该知识点练习题 1-2 道 → 规则判题 → 全部答对：档位+1（间隔翻倍，上限 30 天），
  状态升为 mastered（若 mastery>=0.8）；答错：档位重置 0（间隔 1 天），状态降级 reviewing；
- mastery 由 log_exercise 自动更新（P3 修复的链路），这里只算间隔、写 next_review。

对外接口：
  due_kps(user_id, course)                -> 到期待复习的知识点
  schedule_next(user_id, kp_id, correct)  -> 按结果更新档位并写 next_review，返回下次间隔天数
  run_review_session(...)                 -> 完整复习一轮（CLI/Web 共用）
"""

import random
from datetime import datetime, timedelta

import db
import grader
import learning_pack

# SM-2 间隔序列（天）：答对 1→2→4→8→16→30 封顶
INTERVALS = [1, 2, 4, 8, 16, 30]
MAX_INTERVAL = 30
RESET_INTERVAL = 1  # 答错重置


def schedule_next(user_id: str, kp_id: str, correct: bool,
                  db_path: str | None = None) -> int:
    """按复习结果更新间隔档位 + next_review。返回下次间隔天数。

    答对：档位+1（首次 0→1 天，之后翻倍，封顶 30）；
    答错：档位重置 0（1 天后重试）。
    """
    kp = db.get_kp(user_id, kp_id, db_path) or {}
    idx = int(kp.get("review_interval") or 0)
    if correct:
        days = INTERVALS[idx]  # 当前档位对应的间隔（0→1天, 1→2天, 2→4天...）
        idx = min(idx + 1, len(INTERVALS) - 1)  # 答对后档位前进
    else:
        idx = 0
        days = RESET_INTERVAL
    # 更新档位 + 下次复习日期（upsert_kp 不覆盖 review_interval，需单独写）
    next_review = (datetime.now() + timedelta(days=days)).date().isoformat()
    db.upsert_kp(user_id, {"kp_id": kp_id, "next_review": next_review}, db_path)
    with db._conn(db_path) as c:
        c.execute("UPDATE knowledge_points SET review_interval=? WHERE user_id=? AND kp_id=?",
                  (idx, user_id, kp_id))
    return days


def due_kps(user_id: str, course: str = "python",
            db_path: str | None = None) -> list[dict]:
    """到期待复习的知识点（next_review 非空且到期），附带 title。"""
    due = db.review_due(user_id, db_path)
    graph = learning_pack.load_graph(course)
    titles = {kp["kp_id"]: kp["title"] for kp in graph["knowledge_points"]}
    for k in due:
        k["title"] = titles.get(k["kp_id"], k["kp_id"])
    return due


def exercises_for_review(kp_id: str, course: str = "python",
                         n: int = 2) -> list[dict]:
    """取该知识点练习题用于复习。固定随机种子保证同 kp 每次复习题序稳定。"""
    exs = learning_pack.exercises_by_kp(course).get(kp_id, [])
    random.seed(kp_id)
    return exs[:n]


def run_review_session(user_id: str, course: str = "python",
                       ask_user=None, db_path: str | None = None) -> dict:
    """完整复习一轮：列出到期知识点 → 逐个复习 → 更新间隔。

    ask_user(prompt) -> 用户答案。返回统计。
    """
    due = due_kps(user_id, course, db_path)
    if not due:
        return {"reviewed": 0, "correct": 0, "wrong": 0, "due": 0,
                "message": "没有到期复习的知识点，去学新内容吧。"}

    results = []
    for kp in due:
        print(f"\n===== 复习: {kp['title']}（{kp.get('next_review')} 到期）=====")
        exs = exercises_for_review(kp["kp_id"], course)
        if not exs:
            results.append({"kp_id": kp["kp_id"], "correct": False, "skipped": True})
            continue
        ok_all = True
        for ex in exs:
            print(f"\n--- {ex['prompt']} ---")
            if ex["type"] == "mcq":
                for idx, opt in enumerate(ex["check"]["options"]):
                    print(f"  [{idx}] {opt}")
            answer = ask_user("你的答案: ").strip() if ask_user else ""
            ok, msg = grader.grade(ex, answer)
            db.log_exercise(user_id, ex["ex_id"], kp["kp_id"], ok, answer, msg,
                            db_path=db_path)
            print(("✅ " if ok else "❌ ") + msg)
            if not ok:
                ok_all = False
        days = schedule_next(user_id, kp["kp_id"], ok_all, db_path)
        tag = "答对，间隔翻倍" if ok_all else "答错，重置间隔"
        print(f"  → 下次复习: {_fmt_next(days)}（{tag}）")
        results.append({"kp_id": kp["kp_id"], "correct": ok_all, "next_days": days})

    correct_n = sum(1 for r in results if r.get("correct"))
    return {"reviewed": len(results), "correct": correct_n,
            "wrong": len(results) - correct_n, "due": len(due),
            "results": results}


def _fmt_next(days: int) -> str:
    d = (datetime.now() + timedelta(days=days)).date().isoformat()
    return f"{d}（{days} 天后）"
