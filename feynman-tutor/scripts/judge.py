"""评分 Agent（LLM-as-judge，F 阶段评估闭环，PLAN 22.1 评估行）。

demo 思维是"跑两个 case 看起来不错就发出去"；生产思维是"每类输出有分数线，
达标才可上线，低分有人看"。本模块落地第三环：

- 自动打分：judge_output() 让 LLM 按 4 维 rubric（任务符合/诚实/教学安全/针对性）
  打 1-10 分，阈值 7 分以上算有效回答（行业惯例：业务方只认分数线）；
- 人工只复核低分：bad_case（<7 分 或 评委调用失败）追加进
  evals/review_queue.jsonl——人工不看不计数的全量，只看低分样本；
- 与启发式 rubric 的分工：启发式（scripts/eval_llm.py）抓**确定性灾难失败**
  （泄题关键词/JSON 不可解析），免费可进 CI；评委抓**语义级失败**
  （附和/跑顶/编造得体面），非确定、花钱，只在 live/基准跑。

评委本身也是 LLM，会漂移——所以 judge.md 版本化进 git，prompt/模型变更后
live 门禁会连评委一起重跑校准。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402
import prompts  # noqa: E402

THRESHOLD = 7  # 分数线：>=7 算有效回答
ROOT = Path(__file__).resolve().parent.parent
REVIEW_QUEUE = ROOT / "evals" / "review_queue.jsonl"


def judge_output(task_type: str, context: str, output: str) -> dict:
    """对一条模型输出打分。返回 {"score": int|None, "bad_case": bool, "reasons": str}。

    评委调用/解析失败 → score=None 且 bad_case=True（保守判待复核，
    不把"评委挂了"当成"输出合格"）。
    """
    prompt = prompts.load("judge.md").format(
        task_type=str(task_type)[:300],
        context=str(context)[:1500],
        output=str(output)[:2500],
    )
    try:
        raw = model.chat(
            [{"role": "system", "content": "你是严格的质量评委，只输出合法 JSON。"},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300, caller="judge",
        )
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        score = int(data["score"])
        score = max(1, min(10, score))
        return {"score": score, "bad_case": score < THRESHOLD,
                "reasons": str(data.get("reasons", ""))[:200]}
    except (model.ModelError, KeyError, ValueError, json.JSONDecodeError) as e:
        return {"score": None, "bad_case": True,
                "reasons": f"评委调用/解析失败（保守判待复核）: {type(e).__name__}"}


def append_review_queue(entry: dict, path: Path | None = None) -> None:
    """低分样本追加进人工复核队列（JSONL，一行一样本）。

    与黄金集的区别：黄金集是"已知好坏的自检样本"（防尺子漂移），
    复核队列是"待人工判定的新鲜低分样本"（回收真实 bad case 的入口——
    复核后好样本进黄金集、真问题进修复清单）。
    """
    p = path or REVIEW_QUEUE
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), **entry}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_review_queue(path: Path | None = None) -> list[dict]:
    """读全部待复核样本（人工复核工作台的数据源）。"""
    p = path or REVIEW_QUEUE
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
