# badcase 基准对比报告（同一把尺子实测）

- 生成时间：2026-08-30T22:02:56
- 尺子：scripts/badcase_bench.py（自包含，检查器与评委 prompt 内置，新旧版本同源）
- bad 判定：启发式 rubric 失败 或 LLM 评委 <7 分（行为套件）；实际行为与期望不符（安全套件）

| 版本 | 行为套件 bad | 行为 bad 率 | 安全套件 bad | 安全 bad 率 | 综合 bad 率 |
|---|---|---|---|---|---|
| v1.0(基线) | 0/10 | 0% | 7/10 | 70% | 7/20 = 35% |
| 四模块升级后(HEAD) | 0/10 | 0% | 0/10 | 0% | 0/20 = 0% |

## 明细（bad case 逐条）

### v1.0(基线)
- behavior: 无 bad case ✅
- security: 7 例
  - [鉴权] unauth_users_list: GET /api/users 未带 session → 期望 401 实得 200
  - [实验门禁] lecture_feynman_gate: lecture 组调费曼端点 → 期望 403 实得 200
  - [注入拦截] injection_blocked_at_gate: 提示词窃取注入 → 期望 400（L1 拦截）实得 200
  - [幂等] grade_idempotency: 同 request_id 提交 2 次 → 期望 1 条日志 实得 2 条
  - [校验] malformed_submit_400: answers='not-a-dict' → 期望 400 实得 500
  - [数据泄露] password_hash_leak: 用户列表含 password_hash=True（status=200）
  - [资源滥用] transcript_flood_400: 100 条 transcript → 期望 400（上限校验）实得 200

### 四模块升级后(HEAD)
- behavior: 无 bad case ✅
- security: 无 bad case ✅
