"""学习包质量自检：验证 40 练习题 + 前后测所有题目的标准答案可判。

output/code 题型用 check.sample_answer（标准答案）跑一遍判题；
mcq 题型用 check.answer 判题。全部通过才算学习包合格。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grader  # noqa: E402
import learning_pack  # noqa: E402

# output/code 题的标准答案（与 exercises.jsonl 每题的 check 对应）
# 格式: {ex_id: 标准答案代码}
SAMPLES = {
    # ---- ch1 变量 ----
    "py.ch1.v.1": 'name = "Feynman"\nprint(name)',
    "py.ch1.v.2": "x = 5\ny = 3\nz = x + y",
    "py.ch1.i.1": "print(7/2)\nprint(7//2)",
    "py.ch1.i.2": "r = 2 ** 10",
    "py.ch1.i.3": "print(17 % 5)",
    "py.ch1.s.1": 'print("Hello" + " " + "World")',
    "py.ch1.s.2": 's = "ab" * 3',
    "py.ch1.s.3": 'print(len("Hello"))',
    "py.ch1.s.4": 'print("a\\nb")',
    "py.ch1.m.1": 's = "Hello World"\nprint(s.upper())',
    "py.ch1.m.2": 's = "  hi  "\nclean = s.strip()',
    "py.ch1.m.3": 'print("-".join(["a", "b", "c"]))',
    "py.ch1.b.1": "print(5 > 3)",
    "py.ch1.b.2": "r = (5 > 3) and (2 < 4)",
    "py.ch1.b.3": "print(3 == \"3\")",
    "py.ch1.t.1": "print(int(\"42\") + 8)",
    "py.ch1.t.2": "s = str(42)",
    "py.ch1.t.3": "print(int(3.99))",
    # ---- ch2 列表 ----
    "py.ch2.l.1": "a = [10, 20, 30]\nprint(a[1])",
    "py.ch2.l.2": "a = [1, 2, 3, 4]\nlast = a[-1]",
    "py.ch2.l.3": "a = [1, 2, 3]\nprint(a[0] + a[2])",
    "py.ch2.sl.1": "a = [1, 2, 3, 4, 5]\nprint(a[1:3])",
    "py.ch2.sl.2": "a = [1, 2, 3, 4, 5, 6]\nodd = a[::2]",
    "py.ch2.sl.3": "a = [1, 2, 3, 4]\nprint(a[::-1])",
    "py.ch2.m.1": "a = []\na.append(1)\na.append(2)\na.append(3)",
    "py.ch2.m.2": "a = [3, 1, 2]\na.sort()\nprint(a)",
    "py.ch2.m.3": "a = [1, 2, 3]\npopped = a.pop()",
    "py.ch2.c.1": "r = [x*2 for x in range(4)]",
    "py.ch2.c.2": "print([x for x in range(10) if x % 2 == 0])",
    "py.ch2.d.1": 'd = {"name": "Tom"}\nprint(d["name"])',
    "py.ch2.d.2": 'd = {"a": 1}\nd["b"] = 2',
    "py.ch2.dm.1": 'd = {"a": 1, "b": 2}\nprint(len(d.keys()))',
    "py.ch2.dm.2": 'd = {"a": 1}\nv = d.get("x")',
}


def check_jsonl(items: list[dict], kind: str) -> int:
    fails = 0
    for ex in items:
        if ex["type"] == "mcq":
            ok, msg = grader.grade(ex, str(ex["check"]["answer"]))
        elif ex["type"] == "sql":
            # SQL 题标准答案就是 check.answer_sql
            ok, msg = grader.grade(ex, ex["check"]["answer_sql"])
        else:
            sample = SAMPLES.get(ex["ex_id"])
            if sample is None:
                print(f"[跳过-无样例] {kind} {ex['ex_id']}")
                fails += 1
                continue
            ok, msg = grader.grade(ex, sample)
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"[{status}] {kind} {ex['ex_id']}: {msg[:80]}")
    return fails


def main(course: str = "python"):
    print(f"=== 练习题 (40) [{course}] ===")
    f1 = check_jsonl(learning_pack.load_exercises(course), "ex")
    print("=== 前测 (10) ===")
    f2 = check_jsonl(learning_pack.load_pretest(course), "pre")
    print("=== 后测 (10) ===")
    f3 = check_jsonl(learning_pack.load_posttest(course), "post")
    total = f1 + f2 + f3
    print(f"\n结果: {'全部通过 ✅' if total == 0 else f'{total} 题失败 ❌'}")
    return 1 if total else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("course", nargs="?", default="python")
    args = ap.parse_args()
    sys.exit(main(args.course))
