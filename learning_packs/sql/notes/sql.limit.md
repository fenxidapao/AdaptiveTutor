# LIMIT 限制行数

- `LIMIT n`：只返回前 n 行。
- 常与 ORDER BY 配合取 Top-N：`ORDER BY score DESC LIMIT 3` 取最高分前 3。
- 部分数据库支持 `LIMIT 偏移, 数量` 做分页（sqlite3 支持 `LIMIT n OFFSET m`）。
