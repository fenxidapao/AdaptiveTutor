"""
query_log.py — P1-3 轻量查询 / 接地日志（owner 视角刚需）

每轮对话生成后追加一行 jsonl 到 .query_log.jsonl：
  {ts, question, mode(快速/精确), top_sources:[{rank,source,score}],
   grounded(bool), answer_len}

设计原则（见交接文档 P1-3）：
  - 不碰企业级 observability 栈，纯本地文件，零新依赖。
  - grounded 用 heuristic（不增加 LLM 延迟）：
      答案非空 且 不含「无法回答」→ True；其余 → False。
      （即「模型确实依据资料给出了回答」，而非资料不足时答「无法回答」）
  - 写入失败静默吞掉，绝不因日志问题影响主链路。

落盘路径可由环境变量 QUERY_LOG_PATH 覆盖，默认 ./.query_log.jsonl。
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path

QUERY_LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", ".query_log.jsonl"))


def _is_grounded(answer):
    """接地 heuristic：非空且未声明资料不足 → 视为接地。"""
    if not answer or not answer.strip():
        return False
    if "无法回答" in answer:
        # 模型明确说资料不足 / 拒绝作答 → 未接地
        return False
    return True


def log_query(question, mode, citations, answer, latency_ms=None, rejected=None):
    """
    追加一行查询日志。
    question: 用户问题文本
    mode: "快速" / "精确"
    citations: app.py 拼好的引用结构列表 [{rank,source,score,...}]
    answer: 模型最终回答文本
    latency_ms: 整轮问答耗时（毫秒，可选，看板 #15 用）
    rejected: 是否硬拒答 / 未接地（精确模式才有；快速模式传 None）
    返回: 本条记录的 ts，供 👍/👎 反馈回写定位。
    """
    try:
        top_sources = [
            {
                "rank": c.get("rank"),
                "source": c.get("source"),
                "score": c.get("score"),
            }
            for c in (citations or [])
        ]
        # est_tokens：启发式估算（CJK≈1 token/字，含问题+上下文+回答），非精确计数，
        # 仅作成本量级参考，避免引入真实 tokenizer 依赖。
        est_tokens = round(
            (
                len(question or "")
                + sum(len(c.get("content") or "") for c in (citations or []))
                + len(answer or "")
            )
            / 1.5
        )
        # 毫秒级 ts：反馈回写靠 ts 精确定位，秒级可能同秒撞车
        ts = datetime.now().isoformat(timespec="milliseconds")
        record = {
            "ts": ts,
            "question": question,
            "mode": mode,
            "top_sources": top_sources,
            # 额外信号：回答里是否出现 [n] 引用（含中文模型常用的全角【n】／［n］），
            # 与 grounded 区分，便于事后看模型是否真标了来源
            "has_citation": bool(re.search(r"[\[【［]\d+[\]】］]", answer or "")),
            "grounded": _is_grounded(answer),
            "answer_len": len(answer or ""),
            # ---- 可观测看板（#15，2026-08-25）新增字段 ----
            "latency_ms": latency_ms,
            "est_tokens": est_tokens,
            "rejected": rejected,
            "feedback": None,  # 由 set_feedback 在用户点击 👍/👎 后回写
        }
        with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return ts
    except Exception:
        # 日志失败不影响主功能
        return None


def set_feedback(ts, feedback):
    """
    把 ts 对应日志行的 feedback 字段回写为 👍/👎（就地改行，不重复追加）。

    设计说明：feedback 在写入时恒为 null，用户点击 👍/👎 后才补写，
    避免同一条查询被追加成两行。读改写对单用户本地文件安全；
    某行解析失败跳过、不影响其他行。回写失败静默，不影响主链路。
    """
    if not QUERY_LOG_PATH.exists():
        return
    try:
        lines = QUERY_LOG_PATH.read_text(encoding="utf-8").splitlines()
        out = []
        changed = False
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                out.append(line)
                continue
            if not changed and rec.get("ts") == ts:
                rec["feedback"] = feedback
                changed = True
            out.append(json.dumps(rec, ensure_ascii=False))
        if changed:
            QUERY_LOG_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    except Exception:
        # 回写失败不影响主功能
        pass
