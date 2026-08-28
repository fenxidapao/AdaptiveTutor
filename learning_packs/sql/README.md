# SQL 学习包（learning_packs/sql）

## 内容
- 12 知识点（2 章节）：ch1 SQL 查询基础（SELECT/WHERE/ORDER/DISTINCT/LIMIT）、ch2 聚合与数据操作（聚合/GROUP BY/HAVING/JOIN/INSERT/UPDATE/DELETE）
- 40 道练习（sql 题型 28 + mcq 12），全部可规则判
- 前测 10 题 / 后测 10 题（不同题、同知识点、同难度）
- notes/ 每知识点精编讲解（检索不到时兜底）

## 判题机制（重要）
- 所有 sql 题型基于 **sqlite3 内存预置库**（`sql_grader.py`）：
  - `students(id, name, class_id, score)` + `classes(id, name)` 两表，6 名学生 3 个班
  - 查询题：学生 SQL 结果集 == 参考答案结果集（**列名+行值，顺序敏感**）
  - 写操作题：学生执行后，目标表最终状态 == 参考答案执行后状态
  - 多语句注入被 sqlite3 原生拒绝；结果集上限 200 行
- 纯规则判题，不调 LLM（防漂移，PLAN 10.3）

## 数据来源与许可
- 题目/notes：原创，无版权问题（示例数据为学生成绩表，虚构）
- 讲解材料：经 CourseRAG 检索（source 过滤策略见项目 docs/REFERENCE.md），
  检索不到时用 notes 兜底；如检索到外部材料会在此记录来源+许可

## 对照实验
- ch1（查询基础）用于 lecture 对照模式，ch2（聚合与数据操作）用于 feynman 模式
