"""FeynmanTutor Web 服务（P3）：FastAPI 后端。

设计原则：
1. 复用现有 db / agents / learning_pack / sandbox / grader，不重复实现；
2. 费曼追问改成无状态轮次 API：前端保存 transcript，每轮调 /api/feynman/turn
   拿教练追问，3 轮后调 /api/feynman/summarize 总结盲点（不阻塞、可重试）；
3. 判题仍走规则（grader），LLM 只做讲解/追问/诊断（防漂移，PLAN 10.3）；
4. 多用户：所有 API 带 user_id，前端可选用户。

启动：uvicorn web.app:app --reload --port 8001
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import config  # noqa: E402
import db  # noqa: E402
import grader  # noqa: E402
import learning_pack  # noqa: E402
import model  # noqa: E402
import sandbox  # noqa: E402
from agents import assessor, diagnostic, feynman, loop, planner, recommender  # noqa: E402

app = FastAPI(title="FeynmanTutor", description="基于费曼学习法的个性化学习 Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """静态前端禁启发式缓存（C2 运维，2026-08-29 实测发现）：
    StaticFiles 默认无 Cache-Control，Chromium 对旧文件长期不回源验证——
    实验期间热修 app.js 会有学生拿着旧前端继续做题，实验条件被静默撕裂。
    no-cache = 每次导航回源验证，文件小、本地服务，开销可忽略。"""
    response = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# 可观测性：LLM 调用日志 hook（token/耗时，PLAN 8 多 Agent 效率评估）
db.init_db()
model.set_log_hook(lambda **kw: db.log_llm_call(**kw))

WEB_DIR = Path(__file__).resolve().parent / "static"


# ==================== C 阶段：鉴权 / 配额工具（PLAN 18.3） ====================

def _require_auth(session_id: str | None, user_id: str) -> str:
    """EXPERIMENT_AUTH=1 时强制 session 鉴权：session 解析的 user_id 必须与请求一致。
    返回最终 user_id。默认 0 时原样返回（兼容本机演示与既有测试）。"""
    if not config.EXPERIMENT_AUTH:
        return user_id
    if not session_id:
        raise HTTPException(401, "未登录，请先注册/登录")
    uid = db.get_session_user(session_id)
    if uid is None:
        raise HTTPException(401, "会话无效或已过期，请重新登录")
    if uid != user_id:
        raise HTTPException(403, "无权访问该用户数据")
    return uid


def _check_quota(user_id: str) -> None:
    """每用户每日 LLM 配额 + 全局熔断（EXPERIMENT_AUTH=1 时生效）。"""
    if not config.EXPERIMENT_AUTH:
        return
    exceeded, msg = db.quota_exceeded(user_id)
    if exceeded:
        raise HTTPException(429, msg)


def _start_req(session_id: str | None, user_id: str) -> str:
    """业务端点统一入口：鉴权 + 标记当前用户/会话（hook 计费 + session trace）。返回有效 user_id。"""
    uid = _require_auth(session_id, user_id)
    _check_quota(uid)
    model.set_current_user(uid)
    model.set_current_session(session_id)
    return uid


def _require_session(session_id: str | None) -> str:
    """EXPERIMENT_AUTH=1 时要求任意有效会话（管理类端点，如 /api/users 列表）。
    返回会话归属的 user_id；默认 0 时返回空串（兼容本机演示）。"""
    if not config.EXPERIMENT_AUTH:
        return ""
    if not session_id:
        raise HTTPException(401, "未登录，请先注册/登录")
    uid = db.get_session_user(session_id)
    if uid is None:
        raise HTTPException(401, "会话无效或已过期，请重新登录")
    return uid


def _enforce_feynman_group(user_id: str) -> None:
    """C2 自变量硬门禁（EXPERIMENT_AUTH=1）：费曼追问端点仅 feynman 组可用。

    EXPERIMENT.md §3：实验差异 = feynman 组走 /api/feynman/*（先答后讲+追问），
    lecture 组只看 /api/explain 标准讲解。前端按 group 分支只是 UX——
    测量有效性靠这里（lecture 组即使篡改前端也进不了费曼流程）。
    """
    if not config.EXPERIMENT_AUTH:
        return
    group = (db.get_user(user_id) or {}).get("group_name")
    if group != "feynman":
        raise HTTPException(403, "当前学习模式不含费曼追问环节")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    """存活检查（compose healthcheck 依赖，PLAN 19.5）+ RAG 可达状态
    （C2 运维：讲解静默降级 notes 时在这里可见，不再无告警）。"""
    rag = "down"
    try:
        import urllib.request
        urllib.request.urlopen(config.RAG_BASE_URL.rstrip("/") + "/health", timeout=2)
        rag = "up"
    except Exception:  # noqa: BLE001
        pass
    return {"status": "ok", "service": "feynmantutor",
            "rag": rag, "experiment_auth": config.EXPERIMENT_AUTH}


# ==================== 用户 ====================

def _public_user(user: dict) -> dict:
    """返回给前端的用户信息：剥离 password_hash（安全，C1）。"""
    return {k: v for k, v in user.items() if k != "password_hash"}


@app.post("/api/register")
def api_register(payload: dict):
    """注册：自动随机分组（feynman/lecture 均衡分配）+ 建 session。"""
    user_id = (payload.get("user_id") or "").strip()
    password = payload.get("password") or ""
    name = payload.get("name") or ""
    if not user_id or not password:
        raise HTTPException(400, "user_id 和密码必填")
    if len(password) < 4:
        raise HTTPException(400, "密码至少 4 位")
    try:
        user = db.register_user(user_id, password, name=name or None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    sid = db.create_session(user_id)
    return {"session_id": sid, "user": _public_user(user)}


@app.post("/api/login")
def api_login(payload: dict):
    user_id = (payload.get("user_id") or "").strip()
    password = payload.get("password") or ""
    if not user_id or not password or not db.verify_user(user_id, password):
        raise HTTPException(401, "user_id 或密码错误")
    sid = db.create_session(user_id)
    return {"session_id": sid, "user": _public_user(db.get_user(user_id))}


@app.post("/api/logout")
def api_logout(payload: dict):
    sid = payload.get("session_id")
    if sid:
        db.delete_session(sid)
    return {"ok": True}


@app.get("/api/me")
def api_me(session_id: str = Query(None)):
    uid = db.get_session_user(session_id) if session_id else None
    if not uid:
        raise HTTPException(401, "未登录")
    return _public_user(db.get_user(uid))


@app.get("/api/users")
def api_users(session_id: str = Query(None)):
    """全部用户列表（实验统计用）。鉴权 + 剥离 password_hash（P0-2 修复：
    原实现直接返回整行，未登录可读全部用户密码哈希）。"""
    _require_session(session_id)
    return [_public_user(u) for u in db.list_users()]


@app.post("/api/users")
def api_user_upsert(user_id: str = Query(...), name: str = Query(""),
                    session_id: str = Query(None)):
    """建用户/查用户（遗留兼容端点，前端注册已走 /api/register）。
    实验模式下禁用：绕过密码注册会制造无凭据账号并占用分组均衡计数。"""
    if config.EXPERIMENT_AUTH:
        raise HTTPException(403, "实验模式下请走 /api/register 注册")
    return _public_user(db.get_user(user_id, name=name or None))


@app.get("/api/users/{user_id}")
def api_user(user_id: str, session_id: str = Query(None)):
    """单用户信息。鉴权（只能看自己）+ 剥离 password_hash。"""
    _require_auth(session_id, user_id)
    return _public_user(db.get_user(user_id))


# ==================== 学习包 ====================

@app.get("/api/learning-pack/{course}")
def api_learning_pack(course: str):
    graph = learning_pack.load_graph(course)
    return {
        "course": course,
        "title": graph["title"],
        "chapters": graph["chapters"],
        "knowledge_points": [
            {k: v for k, v in kp.items() if k != "notes_file"}
            for kp in graph["knowledge_points"]
        ],
    }


# ==================== 前测/后测 ====================

@app.get("/api/quiz/{course}/{kind}")
def api_quiz(course: str, kind: str, chapter: str | None = None,
             user_id: str = Query("u0")):
    """取一套题（不含答案）。kind: pretest/posttest。"""
    if kind == "pretest":
        qs = learning_pack.load_pretest(course)
    elif kind == "posttest":
        qs = learning_pack.load_posttest(course)
    else:
        raise HTTPException(400, "kind 必须是 pretest/posttest")
    if chapter:
        qs = [q for q in qs if q.get("chapter") == chapter]
    return _strip_answers(qs)


def _strip_answers(qs: list[dict]) -> list[dict]:
    """去掉答案字段（mcq 的 answer、output/code 的 check 细节），防作弊。"""
    out = []
    for q in qs:
        q2 = {k: v for k, v in q.items() if k not in ("check",)}
        if q["type"] == "mcq":
            q2["options"] = q["check"]["options"]
        out.append(q2)
    return out


@app.post("/api/quiz/{course}/{kind}/submit")
def api_quiz_submit(course: str, kind: str,
                    payload: dict):
    """提交一套题。payload: {user_id, session_id?, chapter?, mode?, answers: {ex_id: answer}}"""
    # 先鉴权后校验：实验模式下未登录缺参应 401 而非 400（不向未认证方暴露校验细节）
    session_id = payload.get("session_id")
    user_id = _start_req(session_id, str(payload.get("user_id") or ""))
    if not user_id:
        raise HTTPException(400, "user_id 必填")
    answers = payload.get("answers") or {}  # 空 answers 合法（全错提交）
    if not isinstance(answers, dict):
        raise HTTPException(400, "answers 必须是对象")
    answers = payload["answers"]
    chapter = payload.get("chapter")
    mode = payload.get("mode", "feynman")
    if config.EXPERIMENT_AUTH:
        # C1-3：实验分组由服务端按 group_name 强制，前端传的 mode 无效（防篡改破坏对照）
        mode = db.get_user(user_id).get("group_name") or "feynman"
    if kind == "pretest":
        qs = learning_pack.load_pretest(course)
    else:
        qs = learning_pack.load_posttest(course)
    if chapter:
        qs = [q for q in qs if q.get("chapter") == chapter]

    correct = 0
    details = []
    wrong_kps = []
    for q in qs:
        ans = answers.get(q["ex_id"], "")
        ok, msg = grader.grade(q, ans)
        if ok:
            correct += 1
        else:
            kp_id = q.get("kp_id")
            if kp_id and kp_id not in wrong_kps:
                wrong_kps.append(kp_id)
        db.log_exercise(user_id, q["ex_id"], q.get("kp_id", ""), ok, ans, msg)
        details.append({"ex_id": q["ex_id"], "correct": ok, "feedback": msg})

    score = correct / len(qs) if qs else 0.0
    db.record_assessment(user_id, chapter or "all", kind, mode, score, len(qs),
                         elapsed_seconds=payload.get("elapsed_seconds"))
    # E1 学习回流（PLAN 20.2）：后测提交后驱动回流状态机（外部验证=重测分数）
    reflow = None
    if kind == "posttest":
        reflow = loop.reflow_after_posttest(user_id, chapter or "all", score,
                                            wrong_kps)
    return {"score": score, "correct": correct, "total": len(qs),
            "details": details, "reflow": reflow}


# ==================== 练习判题 ====================

@app.get("/api/exercises/{course}")
def api_exercises(course: str, kp_id: str | None = None):
    """取练习题（不含答案）。"""
    exs = learning_pack.load_exercises(course)
    if kp_id:
        exs = [e for e in exs if e.get("kp_id") == kp_id]
    out = []
    for e in exs[:10]:
        e2 = {k: v for k, v in e.items() if k != "check"}
        if e["type"] == "mcq":
            e2["options"] = e["check"]["options"]
        out.append(e2)
    return out


@app.post("/api/grade")
def api_grade(payload: dict):
    """判一道题。payload: {user_id, session_id?, ex_id, answer}"""
    session_id = payload.get("session_id")
    user_id = _start_req(session_id, str(payload.get("user_id") or ""))
    if not user_id:
        raise HTTPException(400, "user_id 必填")
    ex_id = payload.get("ex_id")
    if not ex_id:
        raise HTTPException(400, "ex_id 必填")
    answer = payload.get("answer", "")
    ex = _find_exercise(ex_id)
    if ex is None:
        raise HTTPException(404, f"题目不存在: {ex_id}")
    ok, msg = grader.grade(ex, answer)
    db.log_exercise(user_id, ex_id, ex.get("kp_id", ""), ok, answer, msg)
    # E2 练习策略切换（PLAN 20.3）：按该 kp 连续失败次数降级 hint→explain→prereq
    strategy = loop.practice_strategy(user_id, ex.get("kp_id", "")) if ex.get("kp_id") else "hint"
    resp = {"correct": ok, "feedback": msg, "ex_id": ex_id,
            "explanation": ex.get("explanation", ""),
            "strategy": strategy}
    if strategy == "prereq":
        resp["prereq_titles"] = loop.prereq_titles(ex["kp_id"])
    return resp


def _find_exercise(ex_id: str) -> dict | None:
    for ex in learning_pack.load_exercises():
        if ex["ex_id"] == ex_id:
            return ex
    for q in learning_pack.load_pretest() + learning_pack.load_posttest():
        if q["ex_id"] == ex_id:
            return q
    return None


# ==================== 讲解 / 费曼追问 ====================

@app.get("/api/explain/{course}/{kp_id}")
def api_explain(course: str, kp_id: str, user_id: str = Query("u0"),
                session_id: str = Query(None)):
    """讲解知识点（RAG 检索 → LLM 过滤 → notes 兜底）。"""
    uid = _start_req(session_id, user_id)
    try:
        text = feynman.explain_kp(uid, kp_id, course)
    except KeyError:
        raise HTTPException(404, f"未知知识点: {kp_id}")
    return {"kp_id": kp_id, "explanation": text}


@app.post("/api/feynman/turn")
def api_feynman_turn(payload: dict):
    """费曼追问一轮。payload: {course, kp_id, user_id?, session_id?, transcript: [{role, content}...]}"""
    session_id = payload.get("session_id")
    uid = _start_req(session_id, str(payload.get("user_id") or "") or "u0")
    _enforce_feynman_group(uid)
    course = payload.get("course") or "python"
    kp_id = payload.get("kp_id") or ""
    transcript = payload.get("transcript", [])
    graph = learning_pack.load_graph(course)
    kp = graph["_by_id"].get(kp_id)
    if kp is None:
        raise HTTPException(404, f"未知知识点: {kp_id or '缺失'}")
    reply = feynman.generate_followup(kp, transcript, uid)
    return {"coach": reply, "transcript": transcript}


@app.post("/api/feynman/summarize")
def api_feynman_summarize(payload: dict):
    """费曼结束，总结盲点。payload: {course, kp_id, transcript, user_id?, session_id?}"""
    session_id = payload.get("session_id")
    uid = _start_req(session_id, str(payload.get("user_id") or "") or "u0")
    _enforce_feynman_group(uid)
    course = payload.get("course") or "python"
    kp_id = payload.get("kp_id") or ""
    transcript = payload.get("transcript", [])
    graph = learning_pack.load_graph(course)
    kp = graph["_by_id"].get(kp_id)
    if kp is None:
        raise HTTPException(404, f"未知知识点: {kp_id or '缺失'}")
    gaps = feynman.summarize_gaps(kp, transcript, uid)
    # 记录讲解次数（费曼环节算讲解一次）
    if transcript:
        feynman._update_kp_after_explain(uid, kp, None)
    return {"gaps": gaps}


# ==================== 诊断 / 路径 / 推荐 ====================

@app.post("/api/diagnose/{user_id}")
def api_diagnose(user_id: str, session_id: str = Query(None)):
    uid = _start_req(session_id, user_id)
    return diagnostic.diagnose(uid)


@app.get("/api/path/{course}/{user_id}")
def api_path(course: str, user_id: str, session_id: str = Query(None)):
    uid = _start_req(session_id, user_id)
    return planner.plan_path(uid, course)


@app.get("/api/recommend/{course}/{user_id}")
def api_recommend(course: str, user_id: str, top_n: int = 5,
                  session_id: str = Query(None)):
    uid = _start_req(session_id, user_id)
    return recommender.recommend(uid, course, top_n=top_n)


# ==================== E 阶段：回流 / 学习队列（PLAN 20） ====================

@app.get("/api/reflow/{course}/{user_id}")
def api_reflow(course: str, user_id: str, session_id: str = Query(None)):
    """回流状态（报告页/首页"继续学习"卡片）：是否在回流、第几轮、薄弱点清单。纯查询。"""
    uid = _start_req(session_id, user_id)
    return loop.reflow_status(uid, None)


@app.get("/api/queue/{course}/{user_id}")
def api_queue(course: str, user_id: str, session_id: str = Query(None)):
    """今日学习队列（E3 掌握度驱动）：到期复习优先 + 已学未掌握按 mastery 升序。纯查询。"""
    uid = _start_req(session_id, user_id)
    return {"course": course, "queue": loop.daily_queue(uid, course),
            "limit": config.DAILY_QUEUE_LIMIT}


# ==================== 报告 / 画像 ====================

@app.get("/api/report/{course}/{user_id}")
def api_report(course: str, user_id: str, session_id: str = Query(None)):
    uid = _start_req(session_id, user_id)
    r = assessor.report(uid, None)
    # Web 实验流的测评 chapter 恒为 'all'（前端提交不带章节），而 assessor.by_chapter
    # 排除 'all'（避免与 CLI 分章数据重复）——柱状图会因此永远为空。此处为 Web 流
    # 合成聚合行，前端图例显示"总体"。
    if not r.get("by_chapter") and (r.get("pre") is not None or r.get("post") is not None):
        r["by_chapter"] = [{"chapter": "all", "mode": None,
                            "pre": r.get("pre"), "post": r.get("post"),
                            "gain_pp": r.get("gain_pp")}]
    return r


@app.get("/api/heatmap/{user_id}")
def api_heatmap(user_id: str, session_id: str = Query(None)):
    """热力图数据：12 知识点掌握度（0~1）+ 状态。纯查询，不产生 LLM 调用。"""
    if config.EXPERIMENT_AUTH:
        uid = _require_auth(session_id, user_id)
    else:
        uid = user_id
    model.set_current_user(uid)
    model.set_current_session(session_id)
    graph = learning_pack.load_graph()
    kps = db.list_kps(user_id)
    by_id = {k["kp_id"]: k for k in kps}
    cells = []
    for kp in graph["knowledge_points"]:
        st = by_id.get(kp["kp_id"], {})
        mastery = st.get("mastery", 0) or 0
        cells.append({
            "kp_id": kp["kp_id"],
            "title": kp["title"],
            "chapter": kp["chapter"],
            "mastery": round(mastery, 2),
            "status": st.get("status", "new"),
            "seen": st.get("seen_count", 0),
            "correct": st.get("correct_count", 0),
            "explain": st.get("explain_count", 0),
        })
    return {"cells": cells}


# 静态资源（前端）
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
