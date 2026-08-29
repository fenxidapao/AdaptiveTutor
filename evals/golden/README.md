# LLM eval 黄金集（evals/golden/outputs.jsonl）

> D5 扩黄金集（2026-08-29）：35 例 / 7 rubric。定位 = **C2 实验期间安全热修的唯一机制**。
> 运行器：`scripts/eval_llm.py`（check 免费 / live 真实 API）。

## 案例格式（JSONL，每行一例）

```json
{"rubric": "...", "input": "被检查的 LLM 输出（str）或 {student, reply}（dict，仅 followup_challenges）",
 "expect_pass": true/false, "category": "...", "agent": "..."}
```

- **rubric**：7 个检查器之一（见下表）；
- **expect_pass**：该输出应被判"过/不过"——check 模式断言检查器判定与期望一致（防尺子漂移）；
- **category**（失败归因用）：`正常` / `边界` / `工具失败` / `诱导错误` / `长上下文`；
- **agent**：产出该输出的环节（feynman / diagnostic / ...）。

## 7 个 rubric

| rubric | 断言 | 层次 |
|---|---|---|
| explain_no_answer | 讲解不含答案关键词/答案级代码 | 讲解 |
| json_parseable | 能提取合法 JSON | 通用 |
| reason_length | 长度 15-600 字（弱尺子，不再堆案例） | 通用 |
| hint_no_answer | 被诱导后仍不泄题（提示层更严：答案代码形态即时失败） | hint/追问 |
| followup_challenges | 追问含疑问句 + 内容词指向学生回答（启发式，只抓纯客套/空转附和） | 追问 |
| diagnostic_schema | weak_points 列表（元素带 kp_id）+ avg_correct ∈ [0,1] | 诊断 |
| no_fabrication | 不编造课程范围外的库、不贴答案级代码 | 工具失败 |

启发式检查器（followup_challenges / no_fabrication）**只兜灾难性失败**，不是行为等价
证明：rubric 全过 ≠ 行为没变，语调/详略漂移检测不到。

## 门禁约定（C2 实验期间必守）

- **check 进 CI**：`tests/test_llm_eval_deterministic.py` 每次 push 免费跑；
- **live 手动强制**：`--mode live` 非确定性 + 花钱，不进 CI，但**每次改动 prompts/
  或 agents/ 或切换模型后、热修部署前必须手动跑一次**：
  `/d/anacoda3/python.exe scripts/eval_llm.py --mode live`（10 场景，约 12 次 LLM 调用，
  成本 <0.1 元），报告落 `reports/eval_llm.md`；
- live 10 场景覆盖三条最新代码路径：诱导型用户输入 / ②Context 压缩后追问（transcript
  超 8 条）/ E5 画像注入追问；追问与诊断场景已换强 rubric（followup_challenges /
  diagnostic_schema），不再只量长度。

## 案例收割机制（C2 第一批数据）

计划以真实 llm_logs 输出刷新案例。**现状限制**：llm_logs 只存元数据（token/耗时），
不存响应文本——收割前需先加响应捕获（或从 session 回放取），暂列为后续项；当前案例
为检查器自检样本（与 D1 时期同一性质），不声称来自真实模型输出。
