# AdaptiveTutor · 多模态自适应教学 Agent

> 把「费曼教学法」做成一个可运行、可评测、可复现的学习系统：
> 学生先讲，Agent 追问找盲点；知识讲解由**多模态教材知识库**检索支撑，回答可回溯到教材段落与原图；
> 前测/后测 + 沙箱规则判题给出硬反馈。

---

## 1. 这是什么

AdaptiveTutor 由两个进程组成，各自负责一件事：

| 服务 | 目录 | 职责 |
|---|---|---|
| **tutor**（前门） | `feynman-tutor/` | 诊断 / 费曼追问 / 前后测 / 沙箱判题 / 学习回流 / Web 看板 |
| **course-rag**（后端） | `course-rag/` | 教材多模态知识库：切分、嵌入、混合检索、精排、引用与接地自检 |

关系：`tutor` 通过 HTTP 调 `course-rag` 的 `/retrieve` 与 `/images`，
拿到带引用的上下文后由自己调用云端模型生成讲解；`course-rag` 不可用时自动回退到学习包 notes。

```
浏览器 ──> tutor(FastAPI+Web, :8001)
              │  HTTP /retrieve  /images
              └──> course-rag(FastAPI, :8010)
                        │
                        ├── Ollama 本地：嵌入(bge-m3) + 精排(bge-reranker-v2-m3)
                        └── 云端 API：生成(deepseek-v4.1-flash) + 裁判(kimi-k2.6)
```

## 2. 为什么生成/裁判上云，嵌入/精排留本地

本机只有 **8 GB 显存**。本地生成 `llama3.1:8b`（约 4.9 GB）+ 本地裁判 `qwen2.5:7b`（约 4.7 GB）
同时加载超过显存；Windows WDDM **不报 OOM**，而是把 CUDA 调用挂住（日志长时间无写入、CPU 仍上涨）。

因此：

- **生成、裁判** → 云端 OpenAI 兼容 API（显存占用归零）
- **嵌入、精排** → 留在本机（`bge-m3` 约 1.2 GB + reranker 约 2.2 GB，可共存）

> ⚠️ **数据边界**：用户问题与检索到的教材正文会发送给中转 API；嵌入阶段不出本机。
> 需要严格「数据不出本机」时，把 `LLM_PROVIDER` / `JUDGE_PROVIDER` 都设为本地。

实测（2026-10-16）：`deepseek-v4.1-flash` 默认开启思维链，在真实 RAG 长提示下 3/3 次把预算烧在
reasoning 上、`content` 为空；**关闭思维链后 4/4 成功、约 2s**。代码默认已关闭。

## 3. 快速开始

### 3.1 前置

- Python 3.11+（tutor）与 3.11/3.13（course-rag）
- 宿主 Ollama 已启动，且已拉取 `bge-m3`；精确模式另需 `bge-reranker-v2-m3` 权重
- 一个 OpenAI 兼容的中转 API key

```bash
ollama serve
ollama pull bge-m3
```

### 3.2 配置

根目录创建 `.env`（**不随仓库分发**）：

```ini
# course-rag 生成/裁判用
LLM_API_KEY=sk-xxxx
# tutor 生成用（可与上同一把）
DEEPSEEK_API_KEY=sk-xxxx
```

### 3.3 启动（Docker）

```bash
docker compose up -d --build
# tutor      -> http://127.0.0.1:8001
# course-rag -> http://127.0.0.1:8010/docs
```

> 首次构建 `course-rag` 需装 CPU 版 torch（reranker 用），镜像较大、耗时较长。

### 3.4 端到端冒烟

```bash
python smoke/adaptive_tutor_smoke.py
```

它会检查：`course-rag` 可达 → `/retrieve` 真返回文档 → `tutor /api/explain` 走通全链路并返回非空讲解。

## 4. 目录

```
AdaptiveTutor/
├── compose.yaml                    # 根编排：course-rag + tutor 两个服务
├── smoke/adaptive_tutor_smoke.py   # 端到端冒烟
├── course-rag/                     # git subtree：多模态知识库检索
└── feynman-tutor/                  # git subtree：费曼教学 Agent
```

两个子目录用 `git subtree` 维护，保留各自完整提交历史与服务边界：

```bash
# 从上游同步（示例）
git subtree pull --prefix=course-rag     <CourseRAG 仓库> main
git subtree pull --prefix=feynman-tutor  <FeynmanTutor 仓库> main
```

## 5. 能力一览

**tutor**
- 前测 → 诊断 → 费曼教学（先答后讲 + 追问找盲点）→ 沙箱规则判题 → 后测 → 效果报告
- 多 Agent 编排、上下文治理（状态快照注入）、三层记忆、学习回流、每日配额与熔断
- 五层安全纵深（注入筛查 / Schema 约束 + 幂等 / 风险分级 / 业务校验 / 全链路审计）
- 168 项自动化测试 + LLM 黄金集 + Playwright E2E

**course-rag**
- 多格式接入（PDF / PPTX / DOCX / MD / TXT）+ 结构感知切分
- 向量 + BM25 混合检索 + RRF 融合 + bge-reranker 精排
- 多模态：图像 caption 入库 + 原图端点 `/images/{path}` + 视觉重排（可选）
- 引用溯源、硬拒答门控、接地自检（LLM 裁判）
- 自实现评估：faithfulness / answer_relevancy / context_precision / context_recall

## 6. 说明

- 两个子项目仍是可独立运行的仓库；本仓库负责把它们编排成一个系统。
- 子目录内的 README 描述各自模块的详细用法与历史评测口径。