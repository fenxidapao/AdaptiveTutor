"""SQL 判题引擎（P2）：sqlite3 内存库执行学生 SQL，与参考答案结果集比对。

设计：
- 统一预置"学校数据库"（students / classes 两张小表），所有 SQL 题共用；
- 查询题（SELECT）：学生结果集 == 参考答案结果集（列名 + 行值，顺序敏感）；
- 写操作题（INSERT/UPDATE/DELETE）：学生执行后，比对目标表最终状态 == 参考答案执行后的状态；
- 防注入：单语句执行（sqlite3 execute 原生禁止多语句），只读操作只允许 SELECT；
- 纯规则，不调 LLM。

对外：run_sql(statement, check) -> (ok, msg)；建库由学习包触发。
"""

import sqlite3
import uuid

# 统一预置库（所有 SQL 题共用，题目依赖这套数据）
SCHEMA_SQL = """
CREATE TABLE classes (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);
CREATE TABLE students (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  class_id INTEGER,
  score INTEGER,
  FOREIGN KEY (class_id) REFERENCES classes(id)
);
"""

SEED_SQL = """
INSERT INTO classes (id, name) VALUES
  (1, '数学一班'), (2, '英语二班'), (3, '计算机三班');
INSERT INTO students (id, name, class_id, score) VALUES
  (1, '张三', 1, 85),
  (2, '李四', 1, 92),
  (3, '王五', 2, 78),
  (4, '赵六', 2, 65),
  (5, '孙七', 3, 95),
  (6, '周八', 3, 88);
"""

MAX_ROWS = 200  # 结果集上限（防 SELECT * 大表 + 笛卡尔积失控）


class SQLError(Exception):
    """SQL 执行失败（语法错/被拒等）。"""


def _new_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_SQL)
    return conn


def _execute(conn: sqlite3.Connection, statement: str) -> tuple[list, list]:
    """执行一条语句，返回 (列名列表, 行值列表)。非 SELECT 返回列名=[]。"""
    stmt = statement.strip().rstrip(";")
    if not stmt:
        raise SQLError("SQL 为空")
    cur = conn.execute(stmt)  # sqlite3 原生禁止多语句（You can only execute one）
    if cur.description is None:  # 非查询（写操作）
        return [], []
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if len(rows) > MAX_ROWS:
        raise SQLError(f"结果集超过 {MAX_ROWS} 行，请检查查询")
    return cols, rows


def _serialize(cols: list, rows: list) -> str:
    """结果集 → 规范化文本（列名+值，可复现比对）。"""
    parts = ["|".join(cols)]
    for r in rows:
        parts.append("|".join(str(v) for v in r))
    return "\n".join(parts)


def run_sql(statement: str, check: dict) -> tuple[bool, str]:
    """判一条 SQL 题。check 结构：
       {answer_sql: 参考答案, expect: 'rows'|'table', table?: 写操作比对的目标表}
    返回 (是否通过, 描述)。
    """
    answer_sql = check.get("answer_sql", "")
    try:
        # 参考答案先在干净库跑，得到期望
        ref_db = _new_db()
        ref_cols, ref_rows = _execute(ref_db, answer_sql)
        # 学生语句在另一个干净库跑（隔离，防学生 SQL 影响后续判题）
        stu_db = _new_db()
        stu_cols, stu_rows = _execute(stu_db, statement)

        if check.get("expect", "rows") == "rows":
            got = _serialize(stu_cols, stu_rows)
            want = _serialize(ref_cols, ref_rows)
            if got == want:
                return True, f"结果一致（{len(stu_rows)} 行）"
            return False, f"结果不符：期望 {want!r}，实际 {got!r}"

        # 写操作：比对目标表最终状态
        table = check.get("table")
        if not table:
            return False, "写操作题必须指定 check.table"
        stu_final = _dump_table(stu_db, table)
        ref_final = _dump_table(ref_db, table)
        if stu_final == ref_final:
            return True, f"表 {table} 状态一致"
        return False, f"表 {table} 状态不符：期望 {ref_final!r}，实际 {stu_final!r}"
    except SQLError as e:
        return False, f"SQL 执行失败：{e}"
    except sqlite3.Error as e:
        return False, f"SQL 错误：{e}"


def _dump_table(conn: sqlite3.Connection, table: str) -> str:
    try:
        cols, rows = _execute(conn, f"SELECT * FROM {table} ORDER BY rowid")
        return _serialize(cols, rows)
    except (SQLError, sqlite3.Error) as e:
        return f"<表读取失败: {e}>"


# 供 grader 调用的统一入口（grader 只认 grade()，这里提供适配）
def grade_sql(exercise: dict, student_answer: str) -> tuple[bool, str]:
    """grader 的 sql 题型入口。"""
    return run_sql(student_answer, exercise.get("check", {}))
