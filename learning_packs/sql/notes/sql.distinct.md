# DISTINCT 去重

- `SELECT DISTINCT 列名`：去除结果中重复的行。
- 常用于"有哪些不同的值"类问题（如不同的班级、不同的城市）。
- 多个列时去重针对整行组合：`SELECT DISTINCT class_id, score` 表示 (class_id, score) 组合去重。
