"""测评 Agent（PLAN 14.7）：前测/后测/效果报告。

前测/后测是同知识点、同难度、**不同题**（P0 验收硬约束，防"记住答案"失真）。
mode 字段区分 lecture（纯讲解，对照组）/ feynman（费曼式，实验组）——PLAN 8 对照实验。
"""

import db
import grader
import learning_pack


def _run_quiz(user_id: str, questions: list[dict], mode: str, kind: str,
              chapter: str, ask_user, db_path: str | None,
              detail: bool = False) -> float | tuple[float, list[str]]:
    """跑一套题（前测或后测）。ask_user(prompt) -> 用户答案（mcq 传索引）。

    detail=True 时返回 (正确率, wrong_kps)：答错的知识点 kp_id 列表
    （E1 回流学习任务清单，PLAN 20.2）。
    """
    score = 0.0
    total = len(questions)
    wrong_kps: list[str] = []
    for i, q in enumerate(questions, 1):
        print(f"\n--- {kind} 第 {i}/{total} 题 ---")
        print(q["prompt"])
        if q["type"] == "mcq":
            for idx, opt in enumerate(q["check"]["options"]):
                print(f"  [{idx}] {opt}")
        ans = ask_user("你的答案: ").strip()
        ok, msg = grader.grade(q, ans)
        if ok:
            score += 1.0
        elif q.get("kp_id") and q["kp_id"] not in wrong_kps:
            wrong_kps.append(q["kp_id"])
        print(("✅ " if ok else "❌ ") + msg)
        # 记录答题日志（前测不记 kp 关联？记上，画像数据源）
        db.log_exercise(user_id, q["ex_id"], q.get("kp_id", ""), ok, ans, msg,
                        db_path=db_path)
    rate = round(score / total, 2) if total else 0.0
    db.record_assessment(user_id, chapter, kind, mode, rate, total, db_path=db_path)
    if detail:
        return rate, wrong_kps
    return rate


def run_pretest(user_id: str, chapter: str | None, mode: str,
                course: str = "python", ask_user=input, db_path: str | None = None) -> float:
    """前测：学之前测基线。返回正确率 0~1。"""
    questions = learning_pack.load_pretest(course)
    if chapter:
        questions = [q for q in questions if q.get("chapter") == chapter]
    print(f"\n===== 前测（{len(questions)} 题，先测你已有的水平）=====")
    return _run_quiz(user_id, questions, mode, "pretest", chapter or "all",
                     ask_user, db_path)


def run_posttest(user_id: str, chapter: str | None, mode: str,
                 course: str = "python", ask_user=input, db_path: str | None = None) -> float:
    """后测：学之后测提升。**不同题、同知识点、同难度**。返回正确率。"""
    questions = learning_pack.load_posttest(course)
    if chapter:
        questions = [q for q in questions if q.get("chapter") == chapter]
    print(f"\n===== 后测（{len(questions)} 题，与上次不同的题）=====")
    return _run_quiz(user_id, questions, mode, "posttest", chapter or "all",
                     ask_user, db_path)


def run_posttest_detail(user_id: str, chapter: str | None, mode: str,
                        course: str = "python", ask_user=input,
                        db_path: str | None = None) -> tuple[float, list[str]]:
    """后测变体：返回 (正确率, 答错知识点列表)——E1 回流任务清单（PLAN 20.2）。"""
    questions = learning_pack.load_posttest(course)
    if chapter:
        questions = [q for q in questions if q.get("chapter") == chapter]
    print(f"\n===== 后测（{len(questions)} 题，与上次不同的题）=====")
    return _run_quiz(user_id, questions, mode, "posttest", chapter or "all",
                     ask_user, db_path, detail=True)


def report(user_id: str, chapter: str | None = None,
           db_path: str | None = None) -> dict:
    """效果报告：找该用户最近一次前测/后测对比，输出提升 pp 与薄弱点覆盖。

    返回 {"pre": 0.4, "post": 0.8, "gain_pp": 40, "weak_points": [...], "covered": [...],
          "by_chapter": [{chapter, mode, pre, post, gain_pp}, ...]}  # by_chapter 供柱状图
    """
    assessments = db.get_assessments(user_id, chapter, db_path)
    pre = [a for a in assessments if a["kind"] == "pretest"]
    post = [a for a in assessments if a["kind"] == "posttest"]
    profile = db.get_profile(user_id, db_path) or {}

    weak_details = db.parse_weak_details(profile)
    weak = [w["kp_id"] for w in weak_details]

    pre_rate = pre[-1]["score"] if pre else None
    post_rate = post[-1]["score"] if post else None

    # 薄弱点按章节过滤：profile 是全局画像，报告指定章节时只显示该章知识点
    if weak and chapter:
        kp_chapter = {}
        for kp in learning_pack.load_graph()["knowledge_points"]:
            kp_chapter[kp["kp_id"]] = kp["chapter"]
        weak = [kp for kp in weak if kp_chapter.get(kp) == chapter]

    covered = []
    if post_rate is not None and weak:
        # 后测命中薄弱点（按知识点覆盖统计）
        post_questions = learning_pack.load_posttest()
        post_by_kp = {}
        for q in post_questions:
            post_by_kp.setdefault(q["kp_id"], []).append(q)
        covered = [kp for kp in weak if kp in post_by_kp]

    # 按章节聚合（柱状图用）：每章最近一次前后测
    by_chapter = []
    for ch in sorted({a["chapter"] for a in assessments
                      if a.get("chapter") and a["chapter"] != "all"
                      and (not chapter or a["chapter"] == chapter)}):
        c_pre = [a for a in assessments if a["chapter"] == ch and a["kind"] == "pretest"]
        c_post = [a for a in assessments if a["chapter"] == ch and a["kind"] == "posttest"]
        p = c_pre[-1]["score"] if c_pre else None
        q = c_post[-1]["score"] if c_post else None
        by_chapter.append({
            "chapter": ch,
            "mode": c_post[-1]["mode"] if c_post else (c_pre[-1]["mode"] if c_pre else None),
            "pre": p, "post": q,
            "gain_pp": round((q - p) * 100, 1) if p is not None and q is not None else None,
        })

    # 按章节过滤 weak_details（与 weak 同步）
    if chapter:
        kp_chapter = {kp["kp_id"]: kp["chapter"]
                      for kp in learning_pack.load_graph()["knowledge_points"]}
        weak_details = [w for w in weak_details if kp_chapter.get(w["kp_id"]) == chapter]

    return {
        "pre": pre_rate, "post": post_rate,
        "gain_pp": round((post_rate - pre_rate) * 100, 1) if pre_rate is not None and post_rate is not None else None,
        "weak_points": weak, "covered": covered,
        "weak_details": weak_details,
        "mode": post[-1]["mode"] if post else None,
        "by_chapter": by_chapter,
    }


def render_report(r: dict) -> str:
    """把 report() 结果渲染成人类可读文本。"""
    lines = ["\n===== 学习效果报告 ====="]
    if r["pre"] is None and r["post"] is None:
        lines.append("还没有前后测数据，先跑 --learn 完整闭环。")
        return "\n".join(lines)
    lines.append(f"  前测正确率: {r['pre'] * 100:.0f}%" if r["pre"] is not None else "  前测: 无数据")
    lines.append(f"  后测正确率: {r['post'] * 100:.0f}%" if r["post"] is not None else "  后测: 无数据")
    if r["gain_pp"] is not None:
        trend = "↑" if r["gain_pp"] > 0 else ("↓" if r["gain_pp"] < 0 else "=")
        lines.append(f"  提升: {r['gain_pp']:+}pp {trend}")
    if r["mode"]:
        lines.append(f"  教学模式: {r['mode']}（{'费曼式' if r['mode']=='feynman' else '纯讲解'}）")
    if r["weak_points"]:
        lines.append(f"  薄弱点: {', '.join(r['weak_points'])}")
        lines.append(f"  薄弱点在后测覆盖: {len(r['covered'])}/{len(r['weak_points'])} 个知识点")
        for w in r.get("weak_details", []):
            ev = ", ".join(w.get("evidence", [])[:3])
            tail = f"（证据: {ev}）" if ev else ""
            reason = f" — {w['reason']}" if w.get("reason") else ""
            lines.append(f"    - {w['kp_id']}{reason}{tail}")
    return "\n".join(lines)
