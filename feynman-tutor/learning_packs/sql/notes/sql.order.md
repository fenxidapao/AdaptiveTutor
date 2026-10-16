# ORDER BY 排序

- `ORDER BY 列名` 排序结果，默认**升序 ASC**。
- 降序要显式写 `DESC`：`ORDER BY score DESC`。
- 可以多列排序：`ORDER BY class_id, score DESC`（先按班级，再按分数降序）。
- 位置：放在 WHERE / GROUP BY 之后。
