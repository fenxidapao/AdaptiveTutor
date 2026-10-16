# UPDATE 更新数据

- `UPDATE 表 SET 列 = 新值 WHERE 条件`：修改符合条件的行。
- **忘记 WHERE 会更新全表**——最常见的数据事故，务必先想好条件。
- SET 里可以引用原值做运算：`SET score = score + 5`。
- 可以一次改多列：`SET score = 90, class_id = 2`。
