"""
grounding.py — 生成接地自检（grounded verifier，2026-08-24 工程化落地 P1）

为什么需要它（见点评文档任务2「幻觉控制 = 强系统提示 + verifier + 选择性拒答」）：
  仅靠 SYSTEM_RULES「只用来源」无法 100% 杜绝模型凭记忆补全。生产 RAG 标配一个
  verifier：生成后用独立裁判判断回答是否真的由检索上下文支撑，未接地则触发硬拒答。

与 eval_local.faithfulness 的关系：
  - 指标口径一致（把回答拆陈述、逐条判 supported、score=被支撑/总数）。
  - 但 eval_local 与 rag_core 互相 import，此处自包含实现，避免循环依赖；
    后续可把两者统一抽到 judge_utils 复用（TODO）。
  - 裁判复用 config.JUDGE_MODEL（qwen2.5:7b，与生成 llama3.1:8b 不同源），温度 0 逼稳定 JSON。
"""
import json
import re

import config
import ollama

_judge_client = ollama.Client(host=config.OLLAMA_BASE_URL)


def _extract_json(text: str):
    """抠第一个合法 JSON 对象；兼容 ```json 围栏。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def verify_grounded(context: str, answer: str) -> dict:
    """
    判定回答是否完全基于上下文。

    返回 {"grounded": bool, "score": float(0~1), "detail": [...]}：
      - score = 被支撑陈述数 / 总陈述数
      - grounded = score >= config.GROUNDED_SCORE_THRESHOLD
    任一异常（裁判不可用/解析失败）一律返回 grounded=False，宁可拒答不可错答。
    """
    if not context or not answer:
        return {"grounded": False, "score": 0.0, "detail": []}

    system = (
        "你是严谨的 RAG 接地裁判。只依据给定的【上下文】判断，严禁使用外部知识。"
        "必须只输出一个 JSON 对象，不要任何额外文字或解释。"
    )
    user = f"""【上下文】
{context}

【回答】
{answer}

任务：
1. 只把【回答】的正文拆成若干条彼此独立的陈述（statements）。
   不要拆分任务指令、格式说明等任何非回答内容。
2. 对每条陈述，判断是否能从【上下文】中直接推导或支持（supported 为 true/false）。
3. 计算 score = 被支撑的陈述数 / 总陈述数（0~1，保留两位小数）。

只输出 JSON，格式：
{{"statements": [{{"statement": "陈述内容", "supported": true}}], "score": 0.0}}"""

    # 裁判最多重试 2 次；失败 → grounded=False（安全侧）。
    parsed = {}
    for _ in range(3):
        try:
            resp = _judge_client.chat(
                model=config.JUDGE_MODEL, messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ], options={"temperature": 0.0},
            )
            parsed = _extract_json(resp["message"]["content"]) or {}
            if parsed:
                break
        except Exception:
            parsed = {}

    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    grounded = score >= config.GROUNDED_SCORE_THRESHOLD
    detail = parsed.get("statements", []) if isinstance(parsed.get("statements"), list) else []
    return {"grounded": grounded, "score": round(score, 3), "detail": detail}
