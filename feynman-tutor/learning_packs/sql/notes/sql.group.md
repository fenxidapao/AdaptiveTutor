# GROUP BY 分组

- `GROUP BY 列名`：把相同值的行归为一组，聚合函数对每组分别计算。
- 典型：`SELECT class_id, COUNT(*) FROM students GROUP BY class_id`——每班一行。
- SELECT 的列要么是分组列，要么是聚合函数（否则无意义）。
- GROUP BY 放在 WHERE 之后（先过滤再分组）。
