"""E 阶段 Loop 工程化（PLAN 20）：学习回流闭环 + 练习策略切换 + 掌握度驱动学习队列。

核心哲学（PLAN 20.1）：Loop = 目标 → 尝试 → **外部验证**（非 LLM 自评）→ 重试/停止。
本模块全部纯规则，不调 LLM（防漂移），阈值集中在 config.py 可调。

对外接口：
  E1 学习回流：
    needs_reflow(pre, post)            后测是否触发回流（纯函数，可单测）
    reflow_after_posttest(user_id, chapter, score, wrong_kps)  后测提交后的回流状态机
    reflow_status(user_id, chapter)    当前回流状态（前端"继续学习"卡片用）
  E2 练习策略切换：
    practice_strategy(user_id, kp_id)  连续失败 ≥2 次 → 教学策略降级
    strategy_hint(exercise, strategy, ...) 按策略生成教学反馈（hint=LLM / explain=讲解 / prereq=前置复习）
  E3 掌握度驱动学习队列：
    daily_queue(user_id, course)       到期复习优先 + 已学未掌握按 mastery 升序
"""

import json

import config
import db
import learning_pack
from agents import scheduler


# ==================== E1 学习回流闭环（PLAN 20.2） ====================


def needs_reflow(pre_rate: float | None, post_rate: float) -> bool:
    """后测是否触发回流：后测 < 阈值 且 提升不足。

    提升不足 = 无前测基线（只看后测）或 后测-前测 < REFLOW_MIN_GAIN。
    设计意图：只对"学了但没学会"（未及格且进步不显著）回流；
    起点极低但进步显著（如 0.1→0.5）的不强留，避免把用户困死在回流里。
    """
    if post_rate >= config.REFLOW_THRESHOLD:
        return False
    if pre_rate is None:
        return True
    return post_rate - pre_rate < config.REFLOW_MIN_GAIN


def reflow_after_posttest(user_id: str, chapter: str, score: float,
                          wrong_kps: list[str],
                          db_path: str | None = None) -> dict:
    """后测提交后调用：驱动回流状态机。返回给前端/CLI 的状态。

    无 open 回流 → 初次后测：needs_reflow 为真则创建第 1 轮回流任务。
    有 open 回流 → 这是重测（外部验证）：
      达标（>= REFLOW_PASS_SCORE）→ completed，闭环结束；
      不达标且未超轮 → 本轮记 failed，开下一轮；
      不达标且超轮 → given_up，防无限循环。

    wrong_kps: 本次后测答错的知识点 kp_id 列表（回流学习任务清单，比 profile 画像更直接）。
    """
    existing = db.get_open_reflow(user_id, chapter, db_path)
    if existing is None:
        # 初次后测：读最近一次前测做提升判断
        pre = None
        for a in reversed(db.get_assessments(user_id, chapter, db_path)):
            if a["kind"] == "pretest":
                pre = a["score"]
                break
        if not needs_reflow(pre, score):
            return {"triggered": False, "reflow": None,
                    "reason": "后测达标或提升显著，无需回流"}
        rec = db.open_reflow(user_id, chapter, score, wrong_kps,
                             round_no=1, db_path=db_path)
        return {"triggered": True, "reflow": rec, "passed": None,
                "round": rec["round"], "weak_kps": wrong_kps,
                "max_rounds": config.REFLOW_MAX_ROUNDS,
                "pass_score": config.REFLOW_PASS_SCORE}

    # 重测（外部验证：重测后测分数，不是 LLM 自评）
    ok = score >= config.REFLOW_PASS_SCORE
    if ok:
        rec = db.settle_reflow(existing["id"], score, "completed", db_path)
        return {"triggered": True, "reflow": rec, "passed": True,
                "round": existing["round"], "weak_kps": wrong_kps,
                "max_rounds": config.REFLOW_MAX_ROUNDS,
                "pass_score": config.REFLOW_PASS_SCORE}
    if existing["round"] >= config.REFLOW_MAX_ROUNDS:
        rec = db.settle_reflow(existing["id"], score, "given_up", db_path)
        return {"triggered": True, "reflow": rec, "passed": False,
                "gave_up": True, "round": existing["round"],
                "weak_kps": wrong_kps, "max_rounds": config.REFLOW_MAX_ROUNDS,
                "pass_score": config.REFLOW_PASS_SCORE}
    # 不达标且未超轮：本轮失败，开下一轮
    db.settle_reflow(existing["id"], score, "failed", db_path)
    rec = db.open_reflow(user_id, chapter, score, wrong_kps,
                         round_no=existing["round"] + 1, db_path=db_path)
    return {"triggered": True, "reflow": rec, "passed": False,
            "round": rec["round"], "weak_kps": wrong_kps,
            "max_rounds": config.REFLOW_MAX_ROUNDS,
            "pass_score": config.REFLOW_PASS_SCORE}


def reflow_status(user_id: str, chapter: str | None = None,
                  db_path: str | None = None) -> dict:
    """当前回流状态（报告页/首页"继续学习"卡片的数据源）。纯查询。"""
    open_rec = db.get_open_reflow(user_id, chapter, db_path)
    reflows = db.list_reflows(user_id, db_path)
    if open_rec is not None:
        try:
            weak = json.loads(open_rec.get("weak_kps") or "[]")
        except ValueError:
            weak = []
        return {"active": True, "round": open_rec["round"],
                "weak_kps": weak, "trigger_score": open_rec["trigger_score"],
                "max_rounds": config.REFLOW_MAX_ROUNDS,
                "pass_score": config.REFLOW_PASS_SCORE,
                "history": reflows}
    if reflows:
        last = reflows[-1]
        return {"active": False, "last_status": last["status"],
                "last_retest_score": last.get("retest_score"),
                "history": reflows}
    return {"active": False, "history": []}


# ==================== E2 练习策略切换（PLAN 20.3） ====================


def practice_strategy(user_id: str, kp_id: str,
                      db_path: str | None = None) -> str:
    """该知识点当前应使用的教学策略：hint / explain / prereq。

    规则（PLAN 20.3）：同一知识点**连续失败** ≥2 次 → 策略降级：
      连续失败 1 次  → hint   （方向提示，LLM 生成，不给答案）
      连续失败 2 次  → explain（标准讲解 + 对比举例）
      连续失败 ≥3 次 → prereq （建议复习前置知识点，图谱 prerequisites）
    连续失败从最近一次答题往前数；最近答对则回到 hint。
    纯规则可单测；切换后是否有效 = 规则判题（外部验证）。
    """
    logs = db.get_exercise_logs(user_id, kp_id, db_path)
    fails = 0
    for log in reversed(logs):
        if log["correct"]:
            break
        fails += 1
    if fails >= 3:
        return "prereq"
    if fails == 2:
        return "explain"
    return "hint"


def prereq_titles(kp_id: str, course: str = "python") -> list[str]:
    """E2 prereq 策略：返回前置知识点的标题（纯规则，图谱 prerequisites 字段）。"""
    graph = learning_pack.load_graph(course)
    kp = graph["_by_id"].get(kp_id)
    if not kp:
        return []
    prereqs = kp.get("prerequisites", [])
    titles = []
    for p in prereqs:
        pkp = graph["_by_id"].get(p)
        if pkp:
            titles.append(f"{pkp['title']}（{p}）")
        else:
            titles.append(str(p))
    return titles


# ==================== E3 掌握度驱动学习队列（PLAN 20.4） ====================


def daily_queue(user_id: str, course: str = "python",
                limit: int | None = None,
                db_path: str | None = None) -> list[dict]:
    """今日学习队列：到期复习优先 + 已学未掌握按 mastery 升序。

    规则（PLAN 20.4）：
      1. 到期复习（next_review <= 今天，SM-2）优先，按到期日升序；
      2. 已学（seen_count>0）且 mastery<0.8 的知识点按 mastery 升序（最弱先学）；
      3. 未学过的知识点（seen=0）不进队列（"掌握度驱动"只针对已开始学的）；
      4. 去重、截断 limit（默认 DAILY_QUEUE_LIMIT）。
    """
    limit = limit or config.DAILY_QUEUE_LIMIT
    graph = learning_pack.load_graph(course)
    titles = {kp["kp_id"]: kp["title"] for kp in graph["knowledge_points"]}
    chapters = {kp["kp_id"]: kp["chapter"] for kp in graph["knowledge_points"]}

    due = db.review_due(user_id, db_path)
    due.sort(key=lambda k: k.get("next_review") or "")
    queue = [{
        "kp_id": k["kp_id"],
        "title": titles.get(k["kp_id"], k["kp_id"]),
        "chapter": chapters.get(k["kp_id"], ""),
        "mastery": k.get("mastery") or 0.0,
        "status": k.get("status", "new"),
        "reason": "review",
        "next_review": k.get("next_review"),
    } for k in due]
    in_queue = {k["kp_id"] for k in queue}

    kps = db.list_kps(user_id, db_path)
    weak = [k for k in kps if (k.get("seen_count") or 0) > 0
            and (k.get("mastery") or 0.0) < 0.8 and k["kp_id"] not in in_queue]
    weak.sort(key=lambda k: (k.get("mastery") or 0.0))
    queue += [{
        "kp_id": k["kp_id"],
        "title": titles.get(k["kp_id"], k["kp_id"]),
        "chapter": chapters.get(k["kp_id"], ""),
        "mastery": k.get("mastery") or 0.0,
        "status": k.get("status", "new"),
        "reason": "weak",
        "next_review": k.get("next_review"),
    } for k in weak]

    return queue[:limit]
