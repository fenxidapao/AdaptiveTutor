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


def log_query(question, mode, citations, answer):
    """
    追加一行查询日志。
    question: 用户问题文本
    mode: "快速" / "精确"
    citations: app.py 拼好的引用结构列表 [{rank,source,score,...}]
    answer: 模型最终回答文本
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
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "mode": mode,
            "top_sources": top_sources,
            # 额外信号：回答里是否出现 [n] 引用（含中文模型常用的全角【n】／［n］），
            # 与 grounded 区分，便于事后看模型是否真标了来源
            "has_citation": bool(re.search(r"[\[【［]\d+[\]】］]", answer or "")),
            "grounded": _is_grounded(answer),
            "answer_len": len(answer or ""),
        }
        with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 日志失败不影响主功能
        pass
