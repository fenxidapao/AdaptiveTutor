"""规则判题器（PLAN 14.4）：纯规则，不调 LLM（防漂移，PLAN 10.3）。

题型分支：
  'output' : 跑代码，stdout.strip() == check['expect_stdout'].strip()
  'code'   : 跑附加测试（check['tests'] 断言列表）
  'mcq'    : 学生选择索引 == check['answer']
  'sql'    : sqlite3 内存库执行学生 SQL，与参考答案结果集比对（P2 SQL 学习包）

grade() 返回 (通过: bool, 结果描述: str)。
"""

from typing import Any

import sandbox
import sql_grader


def _build_test_code(user_code: str, tests: list[str]) -> str:
    """把学生代码 + 断言拼成一段可执行代码。"""
    lines = ["# ---- 学生代码 ----", user_code, "", "# ---- 判题断言 ----"]
    lines.extend(tests)
    return "\n".join(lines)


def grade(exercise: dict, student_answer: str) -> tuple[bool, str]:
    """判一道题。

    student_answer 语义随题型变化：
      output/code: 学生提交的 Python 代码（字符串）
      sql: 学生提交的 SQL 语句（字符串）
      mcq: 学生选择的选项索引（'0'..'n'，字符串）
    """
    ex_type = exercise.get("type", "output")
    check = exercise.get("check", {})

    if ex_type == "mcq":
        options = check.get("options", [])
        answer = check.get("answer")
        try:
            choice = int(student_answer.strip())
        except (ValueError, AttributeError):
            return False, f"请输入选项编号 0-{len(options) - 1}"
        if not (0 <= choice < len(options)):
            return False, f"选项编号超出范围（0-{len(options) - 1}）"
        if choice == answer:
            return True, f"正确：{options[choice]}"
        return False, f"选择 {options[choice]} 不正确"

    # SQL：sqlite3 内存库执行比对（P2 SQL 学习包）
    if ex_type == "sql":
        return sql_grader.grade_sql(exercise, student_answer)

    # output / code：跑代码
    if ex_type == "output":
        expect = check.get("expect_stdout", "")
        result = sandbox.run_python(student_answer)
        if not result["ok"]:
            return False, f"代码运行失败：{result['stderr'][:200] or '退出码非 0'}"
        got = result["stdout"].strip().replace("\r\n", "\n").replace("\r", "\n")
        if got == expect.strip():
            return True, f"输出正确：{got!r}"
        return False, f"输出不符：期望 {expect.strip()!r}，实际 {got!r}"

    if ex_type == "code":
        tests = check.get("tests", [])
        code = _build_test_code(student_answer, tests)
        result = sandbox.run_python(code)
        if not result["ok"]:
            return False, f"测试未通过：{result['stderr'][:200]}"
        return True, f"全部 {len(tests)} 个断言通过"

    return False, f"未知题型：{ex_type}"


def grade_batch(exercises: list[dict], answers: list[str]) -> list[tuple[bool, str]]:
    """批量判题（前测/后测用）。answers 与 exercises 一一对应。"""
    return [grade(ex, ans) for ex, ans in zip(exercises, answers)]


def preview(exercise: dict) -> Any:
    """开发用：直接跑一次标准答案，验证题目本身可判（测试学习包质量）。"""
    if exercise["type"] == "mcq":
        return grade(exercise, str(exercise["check"]["answer"]))
    # output/code 题目在 check 里放一个 sample_answer 便于验证
    sample = exercise["check"].get("sample_answer")
    if sample is None:
        return (False, "无 sample_answer，跳过")
    return grade(exercise, sample)
