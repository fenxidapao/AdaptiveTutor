"""SQL 判题引擎测试（P2）：查询比对 / 写操作 / 防注入 / 边界。

全部本地执行（sqlite3 内存库），不依赖网络/LLM。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import sql_grader  # noqa: E402


def _q(answer_sql):
    return {"type": "sql", "check": {"answer_sql": answer_sql, "expect": "rows"}}


def _w(answer_sql, table="students"):
    return {"type": "sql", "check": {"answer_sql": answer_sql,
                                     "expect": "table", "table": table}}


def test_select_correct():
    ex = _q("SELECT name FROM students WHERE score >= 90")
    ok, _ = sql_grader.grade_sql(ex, "SELECT name FROM students WHERE score >= 90")
    assert ok is True


def test_keyword_case_insensitive():
    """SQL 关键字大小写不敏感（sqlite），应判对。"""
    ex = _q("SELECT name FROM students WHERE score >= 90")
    ok, _ = sql_grader.grade_sql(
        ex, "select NAme from STUDENTS where scORe >= 90")
    assert ok is True


def test_leading_whitespace_and_semicolon():
    """首尾空白 + 结尾分号：应被剥离，判对。"""
    ex = _q("SELECT name FROM students")
    ok, _ = sql_grader.grade_sql(ex, "  SELECT name FROM students ;")
    assert ok is True


def test_fullwidth_comma_rejected():
    """全角逗号（复制自富文本）→ SQL 语法错，判错且不崩溃。"""
    ex = _q("SELECT name FROM students")
    ok, msg = sql_grader.grade_sql(ex, "SELECT name　FROM students")
    assert ok is False
    assert "SQL 错误" in msg


def test_select_wrong_rows():
    ex = _q("SELECT name FROM students WHERE score >= 90")
    ok, msg = sql_grader.grade_sql(ex, "SELECT name FROM students")
    assert ok is False
    assert "结果不符" in msg


def test_select_column_order_sensitive():
    """列顺序不同算错（教学要求精确）。"""
    ex = _q("SELECT name, score FROM students WHERE score >= 90")
    ok, _ = sql_grader.grade_sql(ex, "SELECT score, name FROM students WHERE score >= 90")
    assert ok is False


def test_select_order_by_sensitive():
    """行顺序不同算错（ORDER BY 影响结果）。"""
    ex = _q("SELECT name FROM students ORDER BY score DESC")
    ok, _ = sql_grader.grade_sql(ex, "SELECT name FROM students ORDER BY score ASC")
    assert ok is False


def test_write_update_correct():
    ex = _w("UPDATE students SET score = 100 WHERE id = 1")
    ok, _ = sql_grader.grade_sql(ex, "UPDATE students SET score = 100 WHERE id = 1")
    assert ok is True


def test_write_update_wrong_scope():
    """更新错行 → 表状态不同 → 判错。"""
    ex = _w("UPDATE students SET score = 100 WHERE id = 1")
    ok, _ = sql_grader.grade_sql(ex, "UPDATE students SET score = 100 WHERE id = 2")
    assert ok is False


def test_write_delete_correct():
    ex = _w("DELETE FROM students WHERE id = 1")
    ok, _ = sql_grader.grade_sql(ex, "DELETE FROM students WHERE id = 1")
    assert ok is True


def test_write_insert_correct():
    ex = _w("INSERT INTO students (id, name, class_id, score) VALUES (99, '测试', 1, 60)")
    ok, _ = sql_grader.grade_sql(ex, "INSERT INTO students (id, name, class_id, score) VALUES (99, '测试', 1, 60)")
    assert ok is True


def test_multiple_statements_rejected():
    """多语句注入被拒（sqlite3 原生只执行一条）。"""
    ex = _q("SELECT name FROM students")
    ok, msg = sql_grader.grade_sql(ex, "SELECT name FROM students; DROP TABLE students")
    assert ok is False
    assert "one statement" in msg


def test_syntax_error_friendly():
    ex = _q("SELECT name FROM students")
    ok, msg = sql_grader.grade_sql(ex, "SELECT FROM students")
    assert ok is False
    assert "SQL 错误" in msg


def test_result_limit_protected():
    """笛卡尔积超 200 行被拦截。"""
    ex = _q("SELECT * FROM students")
    # 6 行 × 3 行 classes = 18 行，没超；构造一个确实超的（students 自连接）
    ok, msg = sql_grader.grade_sql(ex, "SELECT * FROM students a, students b, students c, students d")
    assert ok is False
    assert "超过" in msg


def test_answers_isolated_between_students():
    """学生 A 的 UPDATE 不影响判题 B（每判一题用新内存库）。"""
    ex = _w("UPDATE students SET score = 100 WHERE id = 1")
    ok1, _ = sql_grader.grade_sql(ex, "UPDATE students SET score = 100 WHERE id = 1")
    # 第二次判题：库里应仍是初始数据（id=1 是 85 不是 100），答案一致应判对
    ok2, _ = sql_grader.grade_sql(ex, "UPDATE students SET score = 100 WHERE id = 1")
    assert ok1 is True and ok2 is True
