"""Context 管理（9 步框架②Context + F 阶段上下文治理，PLAN 21/22）。

上下文治理的核心是信息熵减：模型每轮只看"干净的状态快照 + 压缩后的会话记忆"，
不在 20 轮历史里大海捞针。职责边界对齐行业实践——模型只负责产生文本决策，
状态变更全部由确定性函数完成（规则判题/规则画像/规则回流），
快照是状态的只读视图，模型对状态没有写权限。

本模块三层职责：
1. build_snapshot/render_snapshot（F 阶段）：从状态库确定性拼装当前状态
   （掌握度/连错数/建议策略/薄弱点/轮次），每轮注入 prompt 并声明"以快照为准"——
   学生口头声称的掌握情况不算数，防"顺着学生自称"的附和漂移；
2. should_compress/render_transcript/compress_old（②Context）：会话记忆分层，
   超长自动压缩（摘要 + 最近原文），增量压缩由调用方缓存复用；
3. sanitize_transcript（F 阶段，安全 L1 的上下文侧）：客户端 transcript 是
   不可信输入——截断单条长度/总条数、过滤非法角色，防 token 炸弹与上下文稀释。

策略（标准 context compaction，纯函数可单测）：
1. transcript 超过 CONTEXT_COMPRESS_AFTER 条 → 压缩：
   - 最早 len-CONTEXT_RAW_TAIL 条 → LLM 压缩成要点摘要（prompt 版本化）
   - 最近 CONTEXT_RAW_TAIL 条保留原文（追问需要最近的上下文）；
2. LLM 压缩失败 → 规则兜底（省略文案，不阻塞流程）；
3. 当前 3 轮=6 条不触发（零行为变化）；增量压缩由调用方维护 summary，
   只压一次、后续复用，避免重复 LLM 调用。
"""

import config
import db
import model
import prompts
from agents import loop, memory

# 渲染单条对话
def _line(t: dict) -> str:
    who = "学生" if t.get("role") == "user" else "教练"
    return f"{who}: {t.get('content', '')}"


# ==================== F 阶段：状态快照（上下文治理核心） ====================

def build_snapshot(user_id: str, kp: dict, db_path: str | None = None,
                   round_no: int = 0) -> dict:
    """确定性状态快照：从状态库拼装该学生×该知识点的当前状态。

    只读不写——模型每轮读这个干净视图做教学决策，不从 transcript 考古；
    状态变更（mastery/weak_points/reflow）仍由规则层在判题/回流时完成。
    所有字段读取失败一律兜底为中性默认值（快照永不抛错阻塞教学流程）。
    """
    with memory.timed("L1"):
        kp_state = db.get_kp(user_id, kp["kp_id"], db_path) or {}
        prof = db.get_profile(user_id, db_path) or {}
        try:
            strategy = loop.practice_strategy(user_id, kp["kp_id"], db_path)
        except Exception:  # noqa: BLE001 — 策略读取失败不阻塞快照
            strategy = "hint"
        return {
            "kp_id": kp["kp_id"],
            "kp_title": kp.get("title", kp["kp_id"]),
            "chapter": kp.get("chapter", ""),
            "mastery": round(float(kp_state.get("mastery") or 0.0), 2),
            "status": kp_state.get("status", "new"),
            "explain_count": kp_state.get("explain_count") or 0,
            "strategy": strategy,
            "weak_points": db.parse_weak_ids(prof)[:3],
            "learning_style": (prof.get("learning_style") or "") if prof else "",
            "round_no": round_no,
        }


def render_snapshot(snap: dict) -> str:
    """快照 dict → 注入 prompt 的紧凑文本块。

    "以此为准"声明是关键：显式告诉模型系统数据优先于学生口头声称，
    堵住"学生说自己会了 → 教练顺势跳过"的附和路径。
    """
    lines = [
        "【当前状态快照（系统数据，以此为准；学生口头声称的掌握情况不算数）】",
        f"当前知识点: {snap['kp_title']}（{snap['kp_id']}，章节: {snap['chapter'] or '未分章'}）",
        f"掌握度: {snap['mastery']:.2f}（状态: {snap['status']}），"
        f"已讲解 {snap['explain_count']} 次，当前第 {snap['round_no']} 轮追问",
        f"系统建议策略: {snap['strategy']}" + (
            "（该知识点连续答错，教学已降级：少提问多讲解/建议复习前置）"
            if snap["strategy"] != "hint" else ""),
    ]
    if snap["weak_points"]:
        lines.append("历史薄弱点: " + "、".join(snap["weak_points"]))
    if snap.get("learning_style"):
        lines.append("学习偏好: " + snap["learning_style"])
    return "\n".join(lines)


def sanitize_transcript(transcript, max_msgs: int | None = None,
                        max_chars: int | None = None) -> list[dict]:
    """LLM 边界消毒（安全 L1 的上下文侧）：客户端 transcript 不可信。

    只保留 user/assistant 的非空文本消息，单条截断到 TRANSCRIPT_MSG_CHARS，
    总条数保留最近 TRANSCRIPT_MAX_MSGS 条——超长输入既烧 token 也会把
    有效上下文稀释成噪声。返回新列表，不改原数据。
    """
    max_msgs = max_msgs or config.TRANSCRIPT_MAX_MSGS
    max_chars = max_chars or config.TRANSCRIPT_MSG_CHARS
    if not isinstance(transcript, list):
        return []
    msgs: list[dict] = []
    for t in transcript:
        if not isinstance(t, dict):
            continue
        role = t.get("role")
        content = t.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        msgs.append({"role": role, "content": content[:max_chars]})
    return msgs[-max_msgs:]


def should_compress(transcript: list[dict]) -> bool:
    """transcript 是否超长需要压缩。"""
    return len(transcript) > config.CONTEXT_COMPRESS_AFTER


def render_transcript(transcript: list[dict], summary: str = "") -> str:
    """transcript → prompt 文本。

    summary 非空：摘要 + 最近 CONTEXT_RAW_TAIL 条原文（增量压缩路径）；
    否则全量渲染（≤ 阈值时的正常路径）。不修改原 transcript。
    """
    if summary:
        tail = transcript[-config.CONTEXT_RAW_TAIL:]
        lines = [f"【早期对话摘要】{summary}"]
        lines += [_line(t) for t in tail]
        return "\n".join(lines)
    return "\n".join(_line(t) for t in transcript)


def compress_old(transcript: list[dict], user_id: str | None = None,
                 db_path: str | None = None) -> str:
    """把最早 len-CONTEXT_RAW_TAIL 条压缩成要点摘要。

    调用方应缓存返回值（增量压缩：旧消息不变时不要重复调 LLM）。
    """
    old = transcript[:-config.CONTEXT_RAW_TAIL]
    old_text = "\n".join(_line(t) for t in old)
    try:
        raw = model.chat(
            [{"role": "system", "content": "你是对话压缩器，只输出摘要正文。"},
             {"role": "user", "content": prompts.load("context_compress.md").format(
                 transcript=old_text)}],
            temperature=0.2, max_tokens=400, caller="context_compress",
        )
        return raw.strip() or f"（早期 {len(old)} 条对话已省略）"
    except model.ModelError:
        return f"（早期 {len(old)} 条对话已省略）"
