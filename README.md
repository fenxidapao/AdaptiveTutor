# FeynmanTutor（费曼导师）

> 基于费曼学习法的个性化学习 Agent：**你当老师讲，Agent 追问找盲点；系统建学习画像、沙箱判题给硬反馈，前测/后测证明学习效果。**
> 教学法（费曼先答后讲）+ 硬评估（前后测/对照实验）+ 工程化（多 Agent/沙箱/可观测性/CI）三合一，真实可落地、可分享给同学使用。

[![GitHub](https://img.shields.io/badge/GitHub-fenxidapao%2FFeynmanTutor-blue)](https://github.com/fenxidapao/FeynmanTutor) · **v1.0**（2026-08 收尾版，139 项自动化测试全绿）

---

## 一、产品：它解决什么问题

编程初学者三大痛点：**"看了会、做不对"**（缺练习与即时反馈）、**"以为自己会了"**（缺主动回忆检验）、**"不知道自己哪里薄弱"**（缺个性化画像）。通用 AI 只答疑不学习，无法回答"我学了 X，效果如何？"。

FeynmanTutor 用一句话回答：**你当老师讲，它追问找盲点，做题沙箱判对错，前后测告诉你进步了多少。**

### 功能闭环（单章流程）

```
前测(基线) → 诊断(画像+证据链) → 费曼教学(先答后讲+追问找盲点) → 沙箱练习(规则判题)
→ 后测(不同题同难度) → 效果报告(提升 pp) + 学习回流(不达标自动重学再测)
```

| 模块 | 说明 |
|---|---|
| **前测/后测** | 不同题、同知识点、同难度（防"记住答案"失真）；后测自动对比出提升百分点 |
| **诊断 Agent** | 读答题记录 → 学习画像（薄弱点带证据链：哪道题错的、错在哪） |
| **费曼导师 Agent** | 角色反转：用户先讲概念，Agent 当学生追问 3 轮暴露盲点（先答后讲是红线，不可配置） |
| **测评 Agent** | 前后测执行 + 效果报告 + 章节对比柱状图 |
| **规划/推荐 Agent** | 画像 × 知识图谱 → 定制学习路径；按薄弱点 + 错题推荐练习 |
| **沙箱判题** | subprocess 沙箱（超时 5s / 输出 64KB / 临时目录）+ **纯规则判题**（输出比对/用例/选择/SQLite 判 SQL），**判题不调 LLM 防漂移** |
| **间隔复习** | SM-2 简化算法：答对间隔 1→2→4→8→16→30 天翻倍，答错重置 1 天 |
| **学习回流（Loop）** | 后测不达标 → 自动回流薄弱点重新学习 → 重测外部验证（上限 2 轮防烧钱） |
| **练习策略切换** | 同一知识点连续失败 → 自动降级 hint → 标准讲解 → 前置知识点复习 |
| **掌握度学习队列** | 每日队列 = 到期复习优先 + 已学未掌握按 mastery 升序 |
| **Web 看板** | 6 步闭环界面 + 知识掌握度热力图（Chart.js） |

### 多课程 / 多用户

- **Python**（沙箱判题，40 规则题）+ **SQL**（sqlite3 内存库判题，40 题）双课程
- 全部状态表带 `user_id`（从第一行代码就是多用户设计）；注册登录 + session 隔离 + 每日 LLM 配额 + 全局熔断（防同学刷爆 key）

---

## 二、技术：工程怎么做的

### 架构

```
用户 ──> 编排层 ──> ①诊断Agent ─┐
             │                   ├─> ②费曼导师Agent（先答后讲+追问）
             │                   ├─> ③测评Agent（前后测/报告）
             │                   ├─> ④规划Agent（P1 路径）
             │                   └─> ⑤推荐Agent（P1 选题）
             ▼                        ▼
      知识检索(RAG)             沙箱执行器(规则判题)
       HTTP→CourseRAG           subprocess 5s/64KB
       + notes 兜底             不调 LLM 防漂移
             ▼                        ▼
         状态库 SQLite（user_id 分区：画像/掌握度/错题/复习/回流）
```

### 关键工程点

| 能力 | 实现 |
|---|---|
| **防漂移判题** | 判题 100% 规则（输出/用例/选择/SQL），LLM 只做讲解/诊断/推荐理由——同一答案两次判题结果必一致 |
| **多 Agent 编排** | 诊断/费曼/测评/规划/推荐 5 Agent 分工，任务解耦、各自可测 |
| **Harness（辔头）** | 沙箱隔离（5s/64KB/临时目录）+ 每用户每日配额 + 全局熔断 + LLM 调用可观测性（token/耗时按环节落库） |
| **Evaluation** | 139 项自动化测试（单元/集成）+ LLM 黄金集（35 例 / 7 rubric，带归因标签）+ Playwright E2E 6 步闭环，全部进 CI 门禁 |
| **Loop 工程化** | 学习回流闭环（后测驱动，重测外部验证）/ 练习策略切换（连续失败降级）/ 掌握度学习队列——纯规则实现，可单测 |
| **Memory 动态画像** | 每次答题后规则增量维护 weak_points（答错追加证据链、掌握后自动移除），跨会话记忆注入费曼追问 |
| **Context 管理** | 费曼长对话超阈值自动压缩（旧轮次 LLM 摘要 + 最近原文保留），为双层循环预留 |
| **部署** | Docker + docker-compose（./data 卷持久化）+ GitHub Actions CI + /health 健康检查 |
| **鉴权安全** | pbkdf2 密码哈希 + uuid4 session 防串台；实验分组由服务端按 group_name 强制（防前端篡改） |

### 技术栈

Python 3 / DeepSeek API（deepseek-v4-flash）/ SQLite / subprocess 沙箱 / FastAPI / Chart.js / Docker / GitHub Actions。
运行时零第三方依赖（标准库 urllib 直连 API），pytest/playwright 仅测试用。

### 项目结构

```
main.py / config.py / model.py     CLI 入口 / 配置 / LLM 封装（重试/兜底/402 预警/日志 hook）
db.py                              SQLite 状态库（user_id 分区 + reflow_logs 回流记录）
sandbox.py / grader.py / sql_grader.py   沙箱执行器 + 规则判题器（Python/SQL）
rag.py                             CourseRAG 检索封装（top-N 全量 → LLM 过滤 → notes 兜底）
agents/                            diagnostic / feynman / assessor / planner / recommender
                                   scheduler（SM-2）/ loop（回流+策略+队列）/ context（压缩）
learning_packs/{python,sql}/       12 知识点 + 40 规则题 + 前后测各 10 + notes 兜底
web/                               FastAPI + 原生前端 + Chart.js 热力图
prompts/                           8 个版本化 prompt（git 可 diff）
tests/                             139 项测试
scripts/                           eval_llm / e2e_flow / analyze_experiment / run_u0_smoke
```

---

## 三、评估：效果怎么证明

### 自动化质量门禁（客观）

- **139 项自动化测试**全绿（单元 + 集成 + LLM 黄金集 + E2E），GitHub Actions CI 强制
- **LLM 输出评估**（D1+D5）：黄金集 35 例 / 7 个 rubric 检查器（讲解不泄题/诱导泄题断言/追问反附和/诊断 JSON schema/工具失败反幻觉等，全例带归因标签），live 10 场景（含诱导输入/长对话压缩/画像注入路径）实测 10/10；门禁约定：check 进 CI，prompt/模型变更后 live 强制手动跑
- **E2E**：Playwright 驱动系统 Edge 实测 6 步闭环（注册→前测→诊断→费曼→后测→报告）

### 真实运行数据（不编造）

`state.db` 的 `assessments` 表自动记录每次前后测。v1.0 收尾实测（`scripts/run_u0_smoke.py`，真实 DeepSeek API，2026-08-25）：

| 章节 | 模式 | 前测 | 后测 | 提升 | 说明 |
|---|---|---|---|---|---|
| ch1 变量/数据类型 | feynman | 100%（5/5） | 100%（5/5） | +0pp | 0 号用户（本人，CS 学生）前测即满分——**天花板效应实证** |
| ch2 列表/字典 | feynman | 100%（5/5） | 100%（5/5） | +0pp | 同上 |

- 查看报告：`python main.py --report python --user u0`
- LLM 用量：`python main.py --usage`（token/耗时按环节聚合）
- 费曼追问链路实测有效：教练能识别"答非所问"并坚持追问（例：`a = [1,2]; b = a; a.append(3)` 后 `b` 是什么），盲点总结可落到具体概念（引用绑定/切片返回新列表/字典共享引用）
- 实测期间 CourseRAG 服务未启动 → 讲解自动走 notes 兜底（降级路径验证，两组同降级不污染对照）

> ⚠️ **数据诚实边界**：0 号用户（本人）是 CS 学生、前测接近满分，单人对照存在天花板效应（上面 +0pp 正是实证），单人数据不能证明教学法增量价值——严谨评估依赖多人随机分组对照，**已于 2026-08-29/30 完成（见下）**。不编造数据是本项目红线。

### 多人对照实验结果（C 阶段，2026-08-30 收数，n=13 真实用户）

- **样本**：13 名真实同学注册（5 feynman / 8 lecture 注册，自动均衡分组），12 人完成闭环，**流失率 8%**；测试/开发账号已按约定剔除（分析报告头展示剔除清单）
- **学习效果（主要结果）**：全部参与者前测均值 **52% → 后测均值 73%（+21pp）**——系统在真实用户身上产生了大幅学习提升
- **组间对比（探索性，样本过小不构成结论）**：

| 口径 | feynman 提升 | lecture 提升 | 组间差 |
|---|---|---|---|
| 全部样本 | +22.0pp（n=5） | +20.0pp（n=7） | +2.0pp |
| 剔除天花板（前测≥90%） | +25.0pp（n=4） | +26.0pp（n=5） | −1.0pp |

- **如实结论**：两种教学法下学员都取得 ~20-26pp 提升；组间差异在两版口径下为 +2.0/−1.0pp——**n=13 的探索性样本无法区分费曼式与纯讲解的增量价值**，与预注册预期（n≥20 才可能有统计意义）一致，实际招募受限已如实记录（docs/EXPERIMENT_LOG.md）
- **实验工程有效性（本项目最硬的部分）**：
  - 分组服务端强制，事后经 llm_logs 核验：feynman 组每人 3-6 次`feynman_followup`调用，lecture 组**零**费曼调用（自变量保真 100%）
  - 预注册假设先于收数写定（docs/EXPERIMENT.md）；天花板（3 例 ≥90%）与快速作答自动标记；作弊/篡改防线经真实数据检验
  - E1 学习回流闭环在生产触发：1 名学员后测未达标 → 自动回流重学 → 重测 +40pp 达标
  - 开闸前 eval 门禁拦下 4 类真实缺陷（教学路径未差异化/密码哈希泄露/容器网络/前端缓存撕裂实验条件），详见 docs/EXPERIMENT_LOG.md

### 对照实验设计要点（收数前预注册）

- 预注册假设："费曼组后测提升显著高于讲解组（α=0.05）"，n≥20；实际 n=13（招募受限），结果定性为探索性
- 排除标准：前测 ≥90% 标记天花板（组间对比报告剔除与否两个版本）
- 作弊检测：答题耗时 <3s/题 标记；mode 由服务端按 group_name 强制，前端无法篡改分组
- 分析脚本：`scripts/analyze_experiment.py`（组间均值/提升 pp/流失率/回流统计，自动剔除测试账号）

---

## 四、快速开始

```bash
# 1. 环境：.env 需含 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
# 2. 自检
python main.py --health

# 3. 完整学习闭环（前测→诊断→费曼→练习→后测→报告）
python main.py --learn python --user u0

# 4. 常用命令
python main.py --report python --user u0      # 效果报告
python main.py --review python --user u0      # 间隔复习（SM-2）
python main.py --queue python --user u0       # 今日学习队列（E3）
python main.py --usage                        # LLM 用量统计
python main.py --learn sql --user u0          # SQL 课程

# 5. Web 看板（浏览器打开 http://127.0.0.1:8001/）
uvicorn web.app:app --host 127.0.0.1 --port 8001

# 6. 测试（139 项，以实测为准）
python -m pytest tests/ -q

# 7. Docker 部署
docker compose up -d --build
```

---

## 五、数据与许可

- 练习题/前后测/notes：原创
- 讲解材料：经 CourseRAG 检索廖雪峰 Python 教程（CC BY-NC-SA 4.0，已核实）作知识底座，不复制原文；检索不可用/不相关时用学习包 notes 兜底
- 详见 `learning_packs/python/README.md`

## 六、文档索引

| 文档 | 内容 |
|---|---|
| [PLAN.md](PLAN.md) | 项目计划书（决策标准/架构/路线图/E 阶段规格） |
| [docs/REFERENCE.md](docs/REFERENCE.md) | 参考项目库（抄什么/看什么） |
| [docs/EXPERIMENT.md](docs/EXPERIMENT.md) | 对照实验设计（预注册假设/排除标准/作弊检测） |
| [docs/HANDOVER-2026-08-23.md](docs/HANDOVER-2026-08-23.md) | 历史交接文档 |

---

**一句话介绍**：「FeynmanTutor——基于费曼学习法的个性化学习 Agent：诊断学习画像、费曼式教学（先答后讲）、沙箱规则判题，前测/后测证明学习效果；多 Agent 编排 + Loop 工程化 + 127 测试全绿，真实可部署、可分享给同学使用。」
