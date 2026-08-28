# DELETE 删除数据

- `DELETE FROM 表 WHERE 条件`：删除符合条件的行。
- **忘记 WHERE 会清空全表**（保留表结构）——与 UPDATE 同理，危险操作。
- 删除后 id 不会自动复用（自增主键继续增长）。
- 只删数据不删表：要删表结构用 DROP TABLE（更危险）。
