"""Context 管理（9 步框架②Context，PLAN 21 Pi 双层循环预留）：费曼长对话压缩。

长对话问题：费曼追问轮数增长（未来内层追问 + 外层直到盲点消除）时，transcript
全量塞进 prompt → token 膨胀 + 上下文稀释（旧轮细节淹没新信息，教练"忘掉"
学生早前讲错什么）。

策略（标准 context compaction，纯函数可单测）：
1. transcript 超过 CONTEXT_COMPRESS_AFTER 条 → 压缩：
   - 最早 len-CONTEXT_RAW_TAIL 条 → LLM 压缩成要点摘要（prompt 版本化）
   - 最近 CONTEXT_RAW_TAIL 条保留原文（追问需要最近的上下文）；
2. LLM 压缩失败 → 规则兜底（省略文案，不阻塞流程）；
3. 当前 3 轮=6 条不触发（零行为变化）；增量压缩由调用方维护 summary，
   只压一次、后续复用，避免重复 LLM 调用。
"""

import config
import model
import prompts

# 渲染单条对话
def _line(t: dict) -> str:
    who = "学生" if t.get("role") == "user" else "教练"
    return f"{who}: {t.get('content', '')}"


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
