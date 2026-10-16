# JOIN 表连接

- `表A JOIN 表B ON 连接条件`：把两张表的行按条件拼起来。
- 典型连接键是外键：`students.class_id = classes.id`。
- 同名列会歧义，加表名前缀：`students.name` vs `classes.name`，可用 `AS 别名` 区分。
- 本题库用的是 INNER JOIN（默认 JOIN），只保留能匹配上的行。
