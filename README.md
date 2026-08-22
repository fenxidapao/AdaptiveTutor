# CourseRAG · 专业课/课本问答助手

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

基于 **本地 Ollama** 的垂直领域 RAG 应用，面向「计算机/数学专业课、课本知识问答」场景。
数据全程不出本机，不依赖任何云端 API。

> 技术栈：自实现 RAG 核心（Ollama 嵌入/生成 + Chroma 向量库 + **自实现 BM25 + RRF 混合检索** + bge-reranker 精排）+ Streamlit
> 定位：个人项目（AI 应用开发方向），在已跑通的 RAG-KnowBot 基础上做工程化改造与垂直落地。
> 注：已彻底去除 LangChain 全家桶，关键组件（切分 / 加载 / 嵌入 / 检索链 / 词面检索）均为自实现；BM25、RRF 融合仅用标准库实现，**零新依赖**，依赖仅剩 `ollama` + `chromadb` + `sentence-transformers`。

---

## 功能

- 📚 多格式课程资料接入（PDF / PPTX / TXT / MD / DOCX；扫描版 PDF 自动告警跳过，不静默空索引）
- ✂️ 递归语义切分（带重叠窗口，避免切断上下文）
- 🔍 **真混合检索（向量全库 + BM25 全库双通道，RRF 融合，reranker 可选）**：向量召回（双塔粗排）与自实现 BM25 词面召回（精确术语如 ACID / 2NF）**各自对全库独立召回** top-50，再用倒数排名融合（RRF）互补两路信号，最后可选 bge-reranker 精排 top-5（单塔交互细排）。BM25 与 RRF 纯标准库实现，零新依赖；正确块即使向量排名靠后，只要词面匹配也能被 BM25 通道独立捞回（对比旧版"向量池内重排"的根本改进，见「评估基线·混合检索」）。
- 📌 **引用溯源（工程化）**：来源带相关性分数与原文偏移，可展开核对，也可直接喂评估；回答正文内联 `[n]` 上标（半角/全角均可），点击可跳转对应来源（P1-2）
- 💬 本地流式问答界面（Streamlit）
- 📊 **本地自实现 RAG 评估**：本机 Ollama 当裁判，四个核心指标测检索/回答质量，零额外依赖
- 🪵 **查询日志（P1-3）**：每轮对话后追加 `.query_log.jsonl`，记录模式 / 来源 / 接地判定 / 引用标记，便于事后回溯检索质量
- ➕ **增量索引 + 去重（P1-4）**：稳定 id（`md5(source::start_index)`）自动去重，上传仅增量 upsert 新文件；按 source 删除单文档，其余不受影响，不再全量重建
- ✅ **核心算法单测（P2-1）**：零依赖 `test_core.py` 覆盖 BM25 排序 / RRF 融合去重 / 稳定 id 去重 / 接地判定 / 日志格式，脱离 chromadb/ollama/streamlit 亦可在任意干净 Python 环境实跑
- ✅ **集成测试 + CI（T9，2026-08-22）**：`test_api.py`（17 用例，fastapi TestClient + mock 外部依赖，无需 Ollama 覆盖全部 API 端点/参数校验/路径穿越防护）+ `test_retrieval.py`（8 用例，检索链路冒烟，无 Ollama 自动 skip）+ GitHub Actions（`.github/workflows/ci.yml`，3.11 矩阵，py_compile + 三套测试）
- 🔒 配置外置 `.env`，模型名/路径/调参不再写死代码

## 目录结构

```
CourseRAG/
├── config.py          # 所有参数集中从 .env 读取（配置外置；含 HYBRID_ENABLED / BM25_TOP_K）
├── rag_core.py        # RAG 核心：加载切分 / 嵌入建库 / 混合检索(hybrid_retrieve) + 重排 / 增量索引(add_documents/delete_by_source) / 拼链
├── rag_search_utils.py# 零依赖纯算法（P2-1 抽出）：分词 / BM25 / RRF 融合 / 稳定 id，供 test_core 独立测试
├── reranker.py        # bge-reranker 跨编码器重排（混合检索的精排段；默认开——难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败）
├── query_log.py       # P1-3 查询日志：每轮落 .query_log.jsonl
├── app.py             # Streamlit 演示界面（含来源展示与内联 [n] 引用跳转）
├── main.py            # FastAPI 服务入口（部署层，与 app.py 共用同一套 rag_core）
├── api_schemas.py     # FastAPI 请求/响应 Pydantic 模型（Swagger 文档自动生成）
├── requirements-api.txt  # API 服务最小依赖（不含 streamlit / torch，供 Docker 镜像）
├── Dockerfile         # 多阶段构建（python:3.11-slim）
├── docker-compose.yml # 一键部署：Ollama + API 双服务
├── .dockerignore      # 镜像排除项（data/、chroma_db/、.env 等）
├── test_core.py       # P2-1 核心算法单测（零依赖，可在任意干净 Python 环境实跑）
├── eval_local.py      # 本地自实现评估（Ollama 当裁判，4 指标，零额外依赖；取代已废弃的 eval.py）
├── eval_data.py       # 评估小样本（question + reference 标准答案）
├── run_ab.py          # 混合检索 A/B 断点续跑 runner（中断后重跑自动续，隔离 reranker 测 BM25 纯贡献）
├── compare_hybrid.py  # 混合检索 vs 纯向量 A/B 对比（整体 + 区别类子集 + 逐题 Δ + 报告）
├── data/              # 课程资料（示例 md + 教材/ 子目录：真实教材 PDF/PPTX 课件）
├── chroma_db/         # 向量库持久化目录（自动生成，已 gitignore）
├── .query_log.jsonl   # 查询日志（自动生成，已 gitignore）
├── .env.example       # 配置模板
└── requirements.txt
```

## 快速开始

```powershell
# 1. 准备环境（创建本地 venv，目录名统一用 venv）
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置
cp .env.example .env

# 3. 启动 Ollama（已装则跳过 pull）
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 4. 运行界面（或双击 start.bat）
streamlit run app.py
# 浏览器打开 http://localhost:8501
# 侧栏确认 data/ 下示例 md 存在 → 点「🔄 全量重建索引（data/ 全部）」→ 问「什么是 ACID？」应带来源回答
# 检索模式：快速=纯向量 top-5（秒级）；精确=向量 top-50 + BM25 融合（+ bge-reranker 精排）top-5（救难 query）
```

## 一键启动

仓库自带 `start.bat`，**双击即用**，会自动按以下顺序探测 Python 解释器：

1. 项目内 `.\venv\Scripts\python.exe`（推荐，你自己建的本地 venv）
2. 环境变量 `%COURSERAG_PYTHON%`（指向任意 `python.exe`）
3. 系统 `PATH` 里的 `python`

```powershell
# 方式一：双击 start.bat
# 方式二：手动（先激活你的 venv）
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

> 首次使用请先 `python -m venv venv` + `pip install -r requirements.txt` 准备好环境，
> 并确保 Ollama 已启动且拉取 `nomic-embed-text` 与 `llama3.1:8b`。
> 脚本已去除所有硬编码路径与中文注释（GBK 编码兼容），在任意 Windows 机器上双击即可运行。

### 干净部署（换机器 / 给别人用 / 从头装）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1:8b
streamlit run app.py
```

依赖已精简为 `ollama` + `chromadb` + `sentence-transformers`（已彻底去除 langchain 全家桶），
`requirements.txt` 显式锁定 `ollama==0.6.2` 与 `chromadb==1.5.9`，降低换机器安装失败的机率。
注意：别人使用时仍需**自备 Ollama 与模型权重**（无法打包进仓库），这是本地 RAG 的固有前提。

## API 服务（FastAPI + Docker 部署）

> 部署层封装（2026-08-20 新增）：把 `rag_core` 的「索引构建 / 检索 / 重排 / 生成」暴露为 RESTful 端点，
> 与 Streamlit 界面**共用同一套核心模块**（`retrieve_and_rerank` / `build_rag_chain`），无重复逻辑，
> 保证「界面看到的来源 = API 返回的来源」。交互式文档：启动后访问 `http://localhost:8000/docs`。

### 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务元信息（模型 / 端点清单） |
| GET | `/health` | 依赖 / Ollama / Reranker 权重自检（复用 `health.py`） |
| POST | `/query` | 问答：检索 + 重排 + 生成，返回回答 + `[n]` 引用来源 + 耗时 |
| POST | `/retrieve` | 仅检索 + 重排（不生成），供前端预检索 / 调试 / 评估 |
| GET | `/documents` | 列出 `data/` 文档与各自块数 |
| POST | `/documents` | 上传课程资料（multipart）→ 保存 + 增量索引（稳定 id 去重） |
| DELETE | `/documents/{name}` | 删除文件 + 只从索引移除该 source 的块 |
| POST | `/index/rebuild` | 全量重建索引（`data/` 全部） |
| GET | `/index/stats` | 向量库块数 / 源文档数 |

> **关于「模型 API 密钥」**：本项目走**本地 Ollama**（数据不出本机是核心卖点），
> 没有云端 API 密钥。真正的运行时配置只有 `OLLAMA_BASE_URL`、模型名、RAG 调参与 `HF_ENDPOINT`
> （见 `.env`）。若未来切云端模型，只需在 `config.py` 增加对应 key 并改 `rag_core` 的调用层。

### 本地运行（与 Streamlit 同机）

```powershell
# 1. 装 API 依赖（与 requirements.txt 版本对齐；若已装过核心依赖则只需 fastapi 三项）
pip install -r requirements-api.txt

# 2. 启动（需 Ollama 已运行且模型已拉取，见「快速开始」）
uvicorn main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000/docs（Swagger UI）
```

### Docker 一键部署（双模式）

> **2026-08-22 变更**：compose 改为 **API 直连宿主 Ollama**（删除了容器内 ollama 服务）。
> 原因：宿主已装 Ollama 且模型齐全，容器内再拉一套要重复下载 ~5GB 模型；
> 且避免 11434 端口冲突。**前置条件：宿主 Ollama 已启动且模型已拉取。**
> 实测验证（Windows Docker Desktop）：`host.docker.internal:11434` 直连宿主 Ollama 成功，
> /health、/query（fast + accurate）、/retrieve 全部正常，容器加载宿主挂载的 993 块索引。

```bash
# 模式一 · slim（默认）：轻量，无 reranker，精确模式=向量+BM25 混合
docker compose up -d --build

# 模式二 · full（完整版）：含 bge-reranker 精排（镜像 +~2GB，首次需下载权重 ~2.2GB）
docker compose --profile full up -d --build

# 验证
curl http://localhost:8000/health   # slim（8000）
curl http://localhost:8001/health   # full（8001）
```

部署说明：
- **双模式通过 `Dockerfile` 的 `WITH_RERANKER` build-arg 控制**（`docker-compose.yml` 已透传）：
  - `api`（默认 slim）：只装 API 最小依赖（`requirements-api.txt`），镜像小、启动快；
    精确模式 = 向量 + BM25 混合（无精排）。
  - `api-full`（`--profile full`）：追加安装 CPU 版 torch + `sentence-transformers`
    （`requirements-rerank.txt`），`RERANKER_ENABLED=true`，精确模式 = 向量 + BM25 + bge-reranker 精排；
    reranker 权重缓存挂 `huggingface_cache` 卷，宿主机已下载过权重则复用，免二次下载。
- **Ollama 连接**：容器内 `OLLAMA_BASE_URL=http://host.docker.internal:11434` 直连宿主；
  Linux 部署时改 `network_mode: host` 或把地址换成宿主 IP。
  ⚠️ 若宿主 Ollama 只监听 127.0.0.1 导致容器连不上：设 `OLLAMA_HOST=0.0.0.0:11434` 重启 Ollama。
- `./data`（课程资料）与 `./chroma_db`（向量库）通过卷挂载进容器，宿主机直接增删文件即可；
  已索引的库在容器内**路径一致**（均为相对 `./data` / `./chroma_db`），无需重建。
- API 启动时自动预热向量库（空库 + 有资料时自动重建），Ollama 未就绪不会阻塞服务启动，
  `/health` 会明确报告缺什么。
- **已知坑（2026-08-22 实测）**：多阶段构建只拷 site-packages，pip 装的可执行脚本不在
  `/usr/local/bin` → Dockerfile CMD 必须用 `python -m uvicorn`（不能用裸 `uvicorn`）。

### 验证（curl 示例）

```bash
# 健康检查
curl http://localhost:8000/health

# 快速问答（纯向量 top-5，秒级）
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"什么是 ACID？","mode":"fast"}'

# 精确问答（向量 + BM25 融合 + reranker 精排，救难 query；本地环境 reranker 默认开）
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"2NF 和 3NF 的区别是什么？","mode":"accurate"}'

# 仅检索（不生成，返回来源列表）
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"死锁的四个必要条件","mode":"accurate"}'

# 上传课程资料（增量索引）
curl -X POST http://localhost:8000/documents -F "file=@./data/我的笔记.md"

# 列出文档 / 查看索引统计
curl http://localhost:8000/documents
curl http://localhost:8000/index/stats
```

`/query` 响应结构（`answer` 中的 `[n]` 对应 `citations` 的 `rank`，可直接渲染跳转）：

```json
{
  "answer": "ACID 是事务的四个特性……[1][2]",
  "mode": "fast",
  "citations": [
    {"rank": 1, "source": "数据库系统笔记.md", "content": "……", "score": 0.31, "start_index": 1024}
  ],
  "latency_ms": 1523,
  "grounded": true
}
```

## 架构（离线构建 / 在线问答）

```mermaid
flowchart TB
    subgraph 离线构建
        A[课程资料<br/>PDF / PPTX / TXT / MD / DOCX] --> B[递归语义切分<br/>800字符 + 150重叠]
        B --> C[嵌入<br/>nomic-embed-text]
        C --> D[(Chroma 向量库<br/>增量索引 + 稳定id去重)]
    end

    subgraph 在线问答
        Q[用户问题] --> E[向量召回 top-50<br/>双塔粗排]
        E --> F[BM25 词面打分<br/>同一批候选]
        F --> G[RRF 倒数排名融合]
        G --> H[bge-reranker 精排 top-5<br/>单塔交互]
        H --> I[拼 Prompt<br/>含 [n] 来源编号]
        I --> J[llama3.1:8b 生成<br/>内联 [n] 引用]
        J --> K[回答 + 来源溯源<br/>分数 / 偏移]
    end

    D -.-> E
    Q -->|快速模式<br/>纯向量 top-5| E
```

```
数据接入 → 切分(800/150) → 嵌入(nomic-embed-text) → Chroma 持久化
                                                    ↓
问题 → 嵌入 → 向量召回 top-50（双塔粗排）
                  ↓
          [自实现 BM25 词面打分（同批候选，标准库实现）]
                  ↓
          [RRF 倒数排名融合 → 向量 + 词面信号互补]
                  ↓
            [bge-reranker 精排 top-5]（单塔交互；现已默认开——难 query 正确块常排 30+，靠 RETRIEVE_K=50 + reranker 纠回）
                  ↓
            拼 Prompt（含 [n] 来源编号）→ llama3.1:8b 生成 → 回答（内联 [n] 可点跳源）+ 来源引用（带分数/偏移）
```

## 相对 RAG-KnowBot 的改造

| 项 | RAG-KnowBot | CourseRAG |
|---|---|---|
| 配置 | 硬编码在文件顶部 | 全部外置 `.env` |
| 模块化 | 单文件 | `config` / `rag_core` / `reranker` 分离 |
| 检索 | 单阶段向量召回 top-5 | 混合检索：向量召回 top-50 + 自实现 BM25+RRF 融合（+ reranker 精排 top-5 可选） |
| 引用 | 折叠面板贴片段 | 结构化：排名 + 相关性分数 + 原文偏移 |
| 评估 | 无 | 本地自实现评估（Ollama 当裁判：faithfulness / answer_relevancy / context_precision / context_recall） |
| 领域 | 通用 demo | 垂直：专业课/课本问答 |
| 模型端点 | 默认本地 | 显式 `base_url`，便于换远程/容器 |

## 评估（本地自实现，W3）

> ⚠️ 关于 RAGAS：曾尝试用 RAGAS，但它 0.1.22 / 0.2.x 在 import 时会拉取
> `langchain_community.chat_models.vertexai`，而 langchain_community 0.3.x 已移除该符号，
> 无法共存（pip 还会把 0.1.22 自动升回 0.2.x）。
> 故**放弃 RAGAS**，改成本机 Ollama 当裁判，**自己实现**四个指标——零额外依赖，反而更贴合
> "AI 应用开发·自己实现关键组件"的工程取舍说明。本项目已彻底去除 langchain，评估脚本仅依赖 Ollama。

纯本地评估，**不碰 HuggingFace**：裁判 LLM 用本机 Ollama（temperature=0 逼稳定输出 JSON），
embedding 用 nomic-embed-text。

```powershell
# 评估脚本只用本机 Ollama 当裁判，无需 ragas/datasets/nest_asyncio/langchain
# 建议先确保 .env 里 RERANKER_ENABLED=false（纯向量召回基线）
python eval_local.py
# 控制台打印四项指标均值，逐题明细（回答/上下文/分数/裁判理由）保存到 eval_results.csv
```

指标含义（四个均由 `eval_local.py` 自实现，LLM-as-judge）：
- `faithfulness`：回答是否真的基于检索到的 context（把回答拆陈述逐条验证，防幻觉，最关键）
- `answer_relevancy`：回答是否切题、是否直接回应问题
- `context_precision`：召回结果里相关内容是否排在前面（理论上 reranker 改善这项；33 样本整体增益≈0，扩到 59 样本全量后 precision 升 +0.120，见下方「59 样本评估基线」）
- `context_recall`：检索是否覆盖了回答问题所需的信息（需 `eval_data.py` 里的 reference）

**对比用法（重点）**：先在 `RERANKER_ENABLED=false` 跑一次得【基线】，开 reranker 再跑一次，
两次 `context_precision` / `faithfulness` 的差值，就是 reranker 在你的数据上价值的硬证据——
这比"拍脑袋说 reranker 有用"有说服力得多。

评估样本在 `eval_data.py`（基于 `data/` 课程笔记 + `data/教材/` 真实教材的知识点，现 100 条），扩文档后往里追加即可。

### 评估基线（21 样本，W4，reranker 关，纯向量召回）

| 指标 | 均值 |
|---|---|
| faithfulness | 0.577 |
| answer_relevancy | 0.933 |
| context_precision | 0.912 |
| context_recall | 0.805 |

**这次 21 样本（对比最初 6 样本）真正暴露的问题，比"指标持平"有价值：**
- 6 样本时 context_precision=1.0 是假象（样本太小、没碰到跨主题题）；扩到 21 样本后降到 0.912，
  原因是**多主题语料下纯向量召回会拉错文档**（例如问 ACID 却召回"树"那篇，答案被迫"无法回答"）。
- `answer_relevancy` 始终 0.93 上下，说明回答切题没问题；短板全在**检索质量**（faithfulness/precision 被拉错文档拖垮）。
- （修正：此前预期"解锁 reranker 后会直接抬升 faithfulness/precision"——**在 33 样本（正确块多在 top-10）场景下被实测推翻**：precision 增益≈0、faithfulness 略降，见下方 reranker OFF vs ON 对比。**但难 query 正确块常排 30+（如"进程和线程的区别"原排 33），此时 reranker 在 RETRIEVE_K=50 候选池里能将其精排回 top-5，故现默认开启。**）
- 备注：裁判是 llama3.1:8b，对回答里的英文术语注释（如 Lost Update）会误判为"超出上下文"，
  faithfulness 绝对值偏保守；但**对比（基线 vs reranker 开）的差值**才是硬证据，绝对值仅作参考。

`eval_results.csv` 为最近一次明细；`eval_results_v1_baseline.csv` 为最初 6 样本（英文 prompt）备份，供对照。

### 评估基线（33 样本，多主题，reranker OFF vs ON）

语料扩到 5 篇（数据库 / 数据结构 / OS / 计网 / 组成原理），评估题扩到 33 条。**OFF 与 ON 均对齐到向量 top-5 上下文**（修正了此前 OFF 喂 10 块 / ON 喂 5 块的不公平对比，见 `rag_core.py` 的 `else` 分支），唯一变量是 reranker 是否开启。

| 指标 | reranker OFF | reranker ON | 差值 |
|---|---|---|---|
| faithfulness | 0.686 | 0.623 | -0.062 |
| answer_relevancy | 0.897 | 0.861 | -0.036 |
| context_precision | 0.730 | 0.723 | -0.007 |
| context_recall | 0.626 | 0.671 | +0.044 |

**实测结论（修正了交接文档的预期）：**
- reranker **未带来整体收益**：context_precision 基本持平（-0.007）。逐题看它把部分题的正确块提前（检查点 +0.675、中序遍历 +0.667、TCP TIME_WAIT +0.417），但也误排另一些题（树术语 -0.478、HTTPS -0.478、Cache -0.417），正负抵消。
- context_recall 是唯一小幅正向（+0.044）：reranker 把向量排名靠后的相关块提到 top-5，使更多题正确块进入上下文。注意——OFF/ON 对齐到 top-5 后，reranker 实际改变了进入 LLM 的 5 块**集合**（不只顺序），所以 recall 会变，并非"重排不动 recall"（该假设仅在 top-N 子集固定时成立）。
- faithfulness / answer_relevancy 轻微负向，主要源于 8b 裁判噪声 + 上下文集合改变后答案波动，非 reranker 直接贡献。
- 根因：在 33 样本（正确块多在 top-10）场景下 reranker 纠错空间小、增益≈0；**但难 query 正确块常排 30+（如"进程和线程的区别"原排 33），扩 RETRIEVE_K=50 + reranker 可将其精排回 top-5（实测 rank 33→top1）**。单塔 reranker 在中文短课程文本上仍有效，关键是候选池要够大。

**处置**：`config.RERANKER_ENABLED` 默认 `true`（难 query 必需）；33 样本整体增益≈0 不代表 reranker 无用——它救的是向量召回失败的难 query（正确块排 30+）。明细见 `eval_results_off_multi.csv` / `eval_results_on_multi.csv`，差值脚本 `compare.py`，结论 `eval_compare.md`。

### 评估基线（59 样本，全量语料，reranker OFF vs ON）

语料扩到 12 篇、评估扩到 59 条后，**已于 2026-08-18 完成全量 OFF/ON 重跑**（此前因时间未执行，见下方「扩展语料说明」）。OFF/ON 仍对齐到向量 top-5 上下文，唯一变量是 reranker；权重已缓存，ON 轮带 `HF_ENDPOINT=https://hf-mirror.com` 加载。

| 指标 | reranker OFF | reranker ON | 差值 |
|---|---|---|---|
| faithfulness | 0.607 | 0.639 | +0.032 |
| answer_relevancy | 0.769 | 0.902 | +0.133 |
| context_precision | 0.624 | 0.744 | +0.120 |
| context_recall | 0.573 | 0.669 | +0.096 |

**实测结论（59 样本全量，修正了 33 样本阶段「增益≈0」的判断）：**
- **四项指标全部正向**，且 context_recall 明显上升（+0.096，相对 +17%）——核心证据。正确块在向量召回中位列 ~33，原本进不了 top-5；`RETRIEVE_K=50` 让其进入召回池，reranker 精排回 top-5 进入 LLM 上下文，召回随之上升。这是交接文档第 6 节「reranker 救难 query」假设的直接证实。
- context_precision 升 +0.120：扩到 12 篇多主题语料后，纯向量召回拉错文档概率更高，reranker 在更大候选池里纠错空间也更大——相比 33 样本时的 -0.007，全量下 reranker 价值清晰显现。
- answer_relevancy 升 +0.133：上下文更准 → 回答更贴题。
- faithfulness 仅 +0.032：提升有限，因忠实度主要取决于「生成是否紧扣所给上下文」，受检索质量影响较小。
- 注意：① 裁判为 8b 自身既生成又打分，faithfulness 绝对值偏保守、有自评价偏差，**看差值而非绝对值**；② precision 收益在题目间分布不均（多数题 Δprec=0，集中在少数「正确块原本排名靠后」的题，如 2NF/+1.0、虚拟内存/+0.604、立即寻址/+0.804），与「召回池内重排」机制一致；③ 代价：reranker 在 CPU 上跑 2.2GB 跨编码器，单题耗时约为 OFF 的 5 倍（~115s vs ~19s）。

明细见 `eval_results_off_new.csv` / `eval_results_on_new.csv`，差值脚本 `compare.py`，结论 `eval_compare.md`。

### 混合检索 vs 纯向量（100 样本，2026-08-22 实测 · v2 真混合检索 · 新语料 697 块）

> ⚠️ **版本说明**：本节为 **v2 真混合检索**（向量全库 + BM25 全库双通道，RRF 融合）在**新语料 697 块**上的实测；
> 裁判为独立模型 **qwen2.5:7b**（与生成模型 llama3.1:8b 不同源，破除自评偏差）。
> v1 旧版（BM25 池内重排，8b 自评，旧语料 25 块）数据见 `eval_compare_hybrid.md` 历史记录。
> **上一版数据（59 题 × 556 块）见 `eval_compare_hybrid.md` 历史记录；本节为 2026-08-22 重测后的最新基线（100 题 × 697 块，含乱码清洗修复）。**

隔离 reranker（`RERANKER_ENABLED=false`）后，对比 **纯向量 top-5** 与 **向量全库 + 自实现 BM25 全库 + RRF 融合 top-5**，量化 BM25 词面召回的纯贡献。

| 指标 | 纯向量 | 混合 | 差值 |
|---|---|---|---|
| faithfulness | 0.72 | 0.81 | +0.09 |
| answer_relevancy | 0.77 | 0.82 | +0.05 |
| context_precision | 0.52 | 0.70 | +0.18 |
| context_recall | 0.58 | 0.75 | +0.16 |

区别类子集（27 题，含 区别/不同/比较/分别/差异/对比/区分/异同）：四项全正向，answer_relevancy **+0.10**、context_recall **+0.15**。

**实测结论（v2 重测，含诚实边界）：**
- **四项指标全部正向，context_recall 由 v1 的 -0.01 转正为 +0.16**：BM25 通道对全库独立召回，把「向量 top-50 之外但词面匹配」的正确块捞回——直接验证了 v2 设计假设（对比 v1 局限：BM25 只在向量池内重排，召回集合不变）。
- 收益同时落在排序层（context_precision **+0.18**）与召回层（context_recall **+0.16**），回答相关性 +0.05。
- **与上一版（59 题 × 556 块：+0.08/+0.10/+0.12/+0.13）对比**：precision/recall 增益**扩大**（+0.12→+0.18、+0.13→+0.16），faithfulness/relevancy 略收窄——教材题（微信小程序/Linux 等术语密集）让 BM25 词面通道更有用武之地；faithfulness 增益收窄部分来自 697 块语料下向量基线本身更强（0.72 vs 旧 0.75 的差异需同语料内看 A/B 差值）。
- **绝对值注意**：context_recall 绝对值（0.58/0.75）低于旧版（0.77/0.89），原因是 100 题里新增了 41 道教材题、正确块更分散，检索难度整体上升——**看 A/B 差值（同裁判同语料）不看绝对值**。
- **数字精度说明**：上方差值四舍五入到 0.01。裁判 qwen2.5:7b 存在小模型判定噪声（实测约 ±0.02~0.03），0.001 精度的绝对值无意义；结论只看方向与粗粒度差值（precision/recall 提升约 +0.16~0.18，远超噪声）。
- 裁判为独立模型 qwen2.5:7b（与生成不同源），仍有小模型判定噪声。
- 语料为 697 块（12 篇课程笔记 + 真实教材/课件 PDF/PPTX），外推到更大语料需谨慎。

明细见 `eval_results_vector.csv` / `eval_results_hybrid.csv`，runner（断点续跑）`run_ab.py`，对比脚本 `compare_hybrid.py`，结论 `eval_compare_hybrid.md`。

> **扩展语料说明（已全量评测）**：`data/` 已扩至 **12 篇**（新增排序算法 / 图与最短路径 / 关系代数与 SQL / 函数依赖与范式 / OS 内存管理 / 计网 IP 与子网路由 / 组成原理指令系统与寻址），`eval_data.py` 扩至 **59 条**，`config.py` 已调 `CHUNK_SIZE=800 / CHUNK_OVERLAP=150 / RETRIEVE_K=50`（候选池扩到 50，让排 30+ 的难 query 正确块也能进池，reranker 才能纠回）。该扩展的 OFF/ON 全量对比**已于 2026-08-18 执行完毕**（见上方「59 样本评估基线」），**最终结论以 59 样本全量为准**；上方 33 样本表作为子集参考保留。重跑命令见交接文档第 4 节（ON 轮须带 `HF_ENDPOINT=https://hf-mirror.com`）。
>
> **2026-08-22 追加**：`eval_data.py` 已扩至 **100 条**（新增 41 条教材题：微信小程序/Streamlit/Linux/算法引论/学习笔记）；乱码清洗白名单修复（剔除拉丁伪字符残留）后语料 697 块。**重测已完成（2026-08-22 17:57）**：上方表格已更新为 100 题 × 697 块的最新基线。
> **2026-08-22 晚间追加**：新增合规公开语料「廖雪峰 Python 教程核心章节」（CC BY-NC-SA 4.0，来源署名见文件头），索引 697→**993 块**（21 源文件）。**注意**：上方 A/B 数字（100 题 × 697 块）为加料前基线，加料后未重测——新旧基线跨语料不可直接比较，重测见路线图。

## 路线图（进行中）

- [x] **P1-1 混合检索**：自实现 BM25 + RRF 融合（向量 + 词面互补），零新依赖（config 新增 `HYBRID_ENABLED` / `BM25_TOP_K`）
- [x] **P1-2 内联引用**：回答正文 `[n]` 上标（半角/全角可点跳源），链复用同一份 docs 消除重复检索与来源不一致
- [x] **P1-3 查询日志**：每轮落 `.query_log.jsonl`（模式 / 来源 / 接地判定 / 引用标记）
- [x] **P1-4 增量索引 + 去重**：稳定 id 自动去重，上传仅增量 upsert，按 source 删单文档（不再全量重建）
- [x] **P2-1 核心算法单测**：零依赖 `test_core.py`，脱离 LLM 栈亦可实跑（7 用例全绿）
- [x] **T9 集成测试 + CI（2026-08-22）**：API 端点测试 17 用例（TestClient + mock，无需 Ollama）+ 检索冒烟 8 用例（无 Ollama 自动 skip）+ GitHub Actions（3.11 矩阵，`python -m pytest test_core.py test_api.py` 全绿）
- [x] 最小可跑闭环 + 配置外置 + 引用溯源
- [x] 接入 bge-reranker 重排并实测（混合检索；代码已验证可启用，难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败，现默认开）
- [x] 引用溯源工程化（分数 + 偏移，可喂评估）
- [x] 本地自实现 RAG 评估（小样本测检索/回答质量，本机 Ollama 当裁判，零额外依赖）
- [x] 中文 prompt + 中文感知切分（W4：prompt 改为中文防幻觉指令；切分器加中文分隔符。指标未明显提升，但暴露了跨主题检索拉错文档的真问题）
- [x] FastAPI 接口 + Docker（2026-08-20 已完成）：`main.py` / `api_schemas.py` / `Dockerfile` / `docker-compose.yml` / `requirements-api.txt`，见「API 服务」章节

## 说明

- **索引管理（P1-4）**：侧栏「🔄 全量重建索引（data/ 全部）」走 `get_vector_store(chunks)` 整体重建；上传文件走增量 upsert（稳定 id 自动去重）；删除单文档只按 source 从索引移除该文档，**其余文档不受影响**，不再全量重建。生产环境建议删除前「移入回收站 + 二次确认」。
- 混合检索默认开（`HYBRID_ENABLED=true`）：精确模式默认即为「向量 + BM25 融合」，即使 `RERANKER_ENABLED=false` 也非纯向量；reranker 已验证可启用：HF 镜像（hf-mirror.com）在本机可达，`bge-reranker-v2-m3` 权重已下载（约 2.2GB）。**默认 `RERANKER_ENABLED=true`**（难 query 正确块常排 30+，需 RETRIEVE_K=50 + reranker 纠回；33 样本整体增益≈0 但救难 query）。启动 app.py 须带 `HF_ENDPOINT=https://hf-mirror.com`（或离线），否则权重加载会去 huggingface.co 超时。
