# HAVING 过滤分组

- `HAVING 条件`：过滤**分组后的结果**（聚合值条件），如 `HAVING COUNT(*) > 1`。
- 和 WHERE 的区别：WHERE 在聚合**前**过滤行，HAVING 在聚合**后**过滤组。
- 位置：GROUP BY 之后、ORDER BY 之前。
- 例：找学生数超过 1 的班——先分组数人数，再 HAVING 筛。
