"""B4：给 Python 学习包 40 题补 explanation 字段（对齐 SQL 包）。

用法：python scripts/backfill_explanation.py
只加字段不改结构；已有 explanation 的题跳过；失败重试 2 次。
生成后人工过一遍（HANDOVER 1.3：LLM 生成 + 人工校对）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model

PACK = Path(__file__).resolve().parent.parent / "learning_packs" / "python"
EX_FILE = PACK / "exercises.jsonl"

PROMPT = """你是编程教学专家。下面是一道 Python 练习题，请用一句话（15-25 字）说明它考察的核心概念，
用于答错后向学生解释"这题考什么"。
题目 JSON：{ex}
只输出解释文本，不要引号、不要前缀、不要 JSON。"""


def main() -> None:
    lines = EX_FILE.read_text(encoding="utf-8").strip().splitlines()
    changed = 0
    skipped = 0
    for i, line in enumerate(lines):
        ex = json.loads(line)
        if ex.get("explanation"):
            skipped += 1
            continue
        text = None
        for _ in range(2):
            try:
                text = model.chat(
                    [{"role": "user", "content": PROMPT.format(ex=json.dumps(ex, ensure_ascii=False))}],
                    temperature=0.2, max_tokens=200,
                ).strip().strip("\"'")
                if text:
                    break
            except model.ModelError:
                text = None
        if not text:
            print(f"[warn] {ex['ex_id']} 生成失败，保留无 explanation")
            continue
        ex["explanation"] = text
        lines[i] = json.dumps(ex, ensure_ascii=False)
        changed += 1
        print(f"  {ex['ex_id']}: {text}")

    EX_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n完成：新增 {changed} 条，已有跳过 {skipped} 条，共 {len(lines)} 题")


if __name__ == "__main__":
    main()
