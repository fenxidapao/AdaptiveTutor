"""费曼导师 Agent（PLAN 14.6，核心）：角色反转教学。

核心流程（先答后讲是红线，不可配置）：
  explain_kp()     讲解知识点（RAG 检索 → LLM 过滤 → notes 兜底；不含练习答案）
  feynman_round()  角色反转：用户当老师讲概念 → Agent 当学生追问 → 3 轮后总结盲点
  hint_only()      答错只给方向提示，不给答案

交互解耦：feynman_round 接受 ask_user 回调（返回用户输入的字符串），
CLI 层传入 input() 包装，测试层可传假输入。
"""

import json
from datetime import datetime
from typing import Callable

import db
import model
import prompts
import rag
from agents import context

# 费曼各环节 prompt 已版本化：prompts/feynman_*.md（D3，PLAN 17.2）


def _load_graph(course: str) -> dict:
    import learning_pack
    return learning_pack.load_graph(course)


def _update_kp_after_explain(user_id: str, kp: dict, db_path: str | None) -> None:
    """讲解后更新 explain_count 与 last_explained。"""
    cur = db.get_kp(user_id, kp["kp_id"], db_path) or {}
    db.upsert_kp(user_id, {
        "kp_id": kp["kp_id"],
        "title": kp.get("title", kp["kp_id"]),
        "chapter": kp.get("chapter", ""),
        "prerequisites": json.dumps(kp.get("prerequisites", []), ensure_ascii=False),
        "explain_count": (cur.get("explain_count") or 0) + 1,
        "last_explained": datetime.now().isoformat(timespec="seconds"),
    }, db_path)


def profile_context(user_id: str, db_path: str | None = None) -> str:
    """历史画像摘要（E5 Memory 跨会话记忆）：供费曼追问/盲点总结注入 prompt。

    新会话不重新诊断也能带上上次的薄弱点与学习偏好（诊断仍是主入口，
    这里是轻量注入）。无画像返回空串（不干扰既有 prompt）。
    """
    prof = db.get_profile(user_id, db_path)
    if not prof:
        return ""
    weak = db.parse_weak_ids(prof)
    style = prof.get("learning_style") or ""
    parts = []
    if weak:
        parts.append("该学生历史薄弱点: " + ", ".join(weak))
    if style:
        parts.append("学习偏好: " + style)
    return "；".join(parts)


def explain_kp(user_id: str, kp_id: str, course: str = "python",
               db_path: str | None = None) -> str:
    """讲解知识点：RAG 检索（LLM 过滤）→ notes 兜底。返回讲解文本。"""
    graph = _load_graph(course)
    kp = graph["_by_id"][kp_id]

    ctx = rag.retrieve_tutor_context(kp_id, course)
    if not ctx["context"]:
        return "（抱歉，暂时没有该知识点的讲解材料，请换个知识点。）"

    if ctx["source"] == "rag":
        print(f"[i] 讲解材料来源: CourseRAG（引用: {', '.join(ctx['used_sources']) or '无'}）")
    else:
        print(f"[i] 讲解材料来源: 学习包 notes 兜底")

    _update_kp_after_explain(user_id, kp, db_path)
    return ctx["context"]


def generate_followup(kp: dict, transcript: list[dict], user_id: str | None = None,
                      db_path: str | None = None, summary: str = "") -> str:
    """基于已有对话生成教练的下一轮提问。

    user_id 传入时注入历史画像（E5 跨会话记忆），让追问针对学生已知薄弱点。
    summary 非空 = 增量压缩结果（②Context：摘要 + 最近原文，由调用方缓存复用）。
    """
    # ②Context 长对话压缩：超阈值时自动压缩旧轮次（增量压缩路径复用 summary）
    if not summary and context.should_compress(transcript):
        summary = context.compress_old(transcript, user_id, db_path)
    lines = context.render_transcript(transcript, summary=summary) or "（刚开始）"
    memory = profile_context(user_id, db_path) if user_id else ""
    user_prompt = prompts.load("feynman_followup.md").format(
        kp_title=kp.get("title", kp["kp_id"]), transcript=lines)
    if memory:
        user_prompt += f"\n\n参考背景：{memory}"
    raw = model.chat(
        [{"role": "system", "content": prompts.load("feynman_system.md")},
         {"role": "user", "content": user_prompt}],
        temperature=0.4, max_tokens=500, caller="feynman_followup",
    )
    return raw.strip() or "你能再举个具体的例子说明吗？"


def summarize_gaps(kp: dict, transcript: list[dict], user_id: str | None = None,
                   db_path: str | None = None, summary: str = "") -> list[str]:
    """3 轮后总结盲点。"""
    if not transcript:
        return []
    if not summary and context.should_compress(transcript):
        summary = context.compress_old(transcript, user_id, db_path)
    lines = context.render_transcript(transcript, summary=summary)
    memory = profile_context(user_id, db_path) if user_id else ""
    user_prompt = prompts.load("feynman_gaps.md").format(
        kp_title=kp.get("title", kp["kp_id"]), transcript=lines)
    if memory:
        user_prompt += f"\n\n参考背景：{memory}"
    try:
        raw = model.chat(
            [{"role": "system", "content": "你只输出合法 JSON。"},
             {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=800, caller="feynman_gaps",
        )
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return [str(x) for x in data.get("gaps", [])][:3]
    except (json.JSONDecodeError, ValueError, model.ModelError):
        return ["（盲点总结失败，请人工查看对话记录）"]


def feynman_round(user_id: str, kp_id: str, course: str = "python",
                  max_rounds: int = 3,
                  ask_user: Callable[[str], str] = input,
                  db_path: str | None = None) -> dict:
    """完整费曼回合：用户讲 → 教练追问，最多 max_rounds 轮 → 总结盲点。

    强制走 API（model.chat 默认不降级）：费曼追问环节 7B 会盲目附和讲错内容，
    API 挂时明确提示"离线模式不支持追问"，而不是降级。

    返回 {'transcript': [...], 'gaps': [...]}。
    """
    graph = _load_graph(course)
    kp = graph["_by_id"][kp_id]
    transcript: list[dict] = []
    summary = ""  # ②Context 增量压缩：只压一次，后续轮次复用

    try:
        for _ in range(max_rounds):
            # 用户当老师讲（第一轮先讲，之后回答追问）
            turn_prompt = "轮到你了：请用你自己的话讲解这个知识点" \
                          "（可以举例子，讲完回车）。" if not transcript \
                          else "请回答教练的追问："
            user_talk = ask_user(turn_prompt).strip()
            if not user_talk:
                print("（跳过本轮）")
                break
            transcript.append({"role": "user", "content": user_talk})

            # 教练追问（最后总结前不再追问）
            if len(transcript) // 2 < max_rounds - 1:
                if not summary and context.should_compress(transcript):
                    summary = context.compress_old(transcript, user_id, db_path)
                coach = generate_followup(kp, transcript, user_id, db_path,
                                          summary=summary)
                print(f"\n[教练] {coach}")
                transcript.append({"role": "assistant", "content": coach})
    except model.ModelError as e:
        print(f"\n[!] 费曼追问不可用（API 故障）：{e}")
        print("    离线模式不支持追问（本地 7B 会盲目附和讲错内容），本次跳过追问。")
        return {"transcript": transcript, "gaps": ["API 不可用，未完成追问"]}

    # 总结盲点
    gaps = summarize_gaps(kp, transcript, user_id, db_path, summary=summary)
    if transcript:
        _update_kp_after_explain(user_id, kp, db_path)
    return {"transcript": transcript, "gaps": gaps}


def hint_only(exercise: dict, result: dict) -> str:
    """答错时只给方向提示，不给答案。"""
    prompt = exercise.get("prompt", "")
    answer = str(result.get("user_answer", ""))[:200]
    feedback = result.get("feedback", "")
    try:
        raw = model.chat(
            [{"role": "system", "content": "你是耐心的编程助教，只给方向提示不给答案。"},
             {"role": "user", "content": prompts.load("feynman_hint.md").format(
                 prompt=prompt, answer=answer, feedback=feedback)}],
            temperature=0.3, max_tokens=300, caller="hint",
        )
        return raw.strip() or "再想想，检查一下你的思路。"
    except model.ModelError:
        return "（提示服务暂不可用）再读一遍题目，检查边界情况。"
