"""
grounding.py — 生成接地自检（grounded verifier，2026-08-24 工程化落地 P1）

为什么需要它（见点评文档任务2「幻觉控制 = 强系统提示 + verifier + 选择性拒答」）：
  仅靠 SYSTEM_RULES「只用来源」无法 100% 杜绝模型凭记忆补全。生产 RAG 标配一个
  verifier：生成后用独立裁判判断回答是否真的由检索上下文支撑，未接地则触发硬拒答。

与 eval_local.faithfulness 的关系：
  - 指标口径一致（把回答拆陈述、逐条判 supported、score=被支撑/总数）。
  - 但 eval_local 与 rag_core 互相 import，此处自包含实现，避免循环依赖；
    后续可把两者统一抽到 judge_utils 复用（TODO）。
  - 裁判走 llm_provider.judge_chat（按 config.JUDGE_PROVIDER 分流），温度 0 逼稳定 JSON。
"""
import json
import re

import config
import llm_provider  # C2：裁判按 config.JUDGE_PROVIDER 分流（local/openai）


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
    # C2（2026-10-16）：此前这里直接 _judge_client.chat 调本地 Ollama，
    # 完全忽略 config.JUDGE_PROVIDER —— config 注释与 README 都宣称可切云端，
    # 但生产 /query 的 grounded 自检实际仍跑本地裁判。现统一走 llm_provider.judge_chat。
    parsed = {}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(3):
        try:
            parsed = _extract_json(llm_provider.judge_chat(messages, temperature=0.0)) or {}
            if parsed:
                break
        except Exception:
            parsed = {}

    detail = parsed.get("statements", []) if isinstance(parsed.get("statements"), list) else []
    # 评审 P2（2026-08-30）：与 eval_local 同口径——分数从明细重算，不信裁判自报。
    # 门控场景取「明细重算」与「裁判自报」的较小值（更保守）：
    #   - 裁判漏列陈述时（P2 复核实测：自报 0.6 但明细 2/2 全 true），漏列的往往是
    #     它没把握的陈述 → 自报分更接近真实，min 避免明细重算虚高放行幻觉；
    #   - 明细与自报一致时二者相等，行为不变。
    # 明细为空（裁判故障）：回退自报分，缺省 0.0（安全侧 → 拒答）。
    recalculated = None
    if detail:
        sup = sum(1 for it in detail if isinstance(it, dict) and it.get("supported") is True)
        recalculated = sup / len(detail)
    try:
        reported = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        reported = 0.0
    score = min(recalculated, reported) if recalculated is not None else reported
    score = max(0.0, min(1.0, score))
    grounded = score >= config.GROUNDED_SCORE_THRESHOLD
    return {"grounded": grounded, "score": round(score, 3), "detail": detail}
