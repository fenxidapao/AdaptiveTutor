# 字典方法

- `d.keys()` 返回所有键，`d.values()` 返回所有值，`d.items()` 返回 (键, 值) 对。
- `d.get(k)` 取值，键不存在返回 `None` 而不是报错；`d.get(k, 默认值)` 可给默认值。
- `d.update({...})` 批量更新；`k in d` 判断键是否存在。
- 遍历常用 `for k, v in d.items():`。
