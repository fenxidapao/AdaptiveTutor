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
- 🔍 **真混合检索（向量全库 + BM25 全库双通道，RRF 融合，reranker 可选）**：向量召回（双塔粗排）与自实现 BM25 词面召回（精确术语如 ACID / 2NF）**各自对全库独立召回** top-50，再用倒数排名融合（RRF）互补两路信号（融合候选池 `RERANK_CANDIDATE_K`=16，供 reranker 精排），最后可选 bge-reranker 精排 top-8（单塔交互细排）。BM25 与 RRF 纯标准库实现，零新依赖；正确块即使向量排名靠后，只要词面匹配也能被 BM25 通道独立捞回（对比旧版"向量池内重排"的根本改进，见「评估基线·混合检索」）。
- 🏷️ **教材级元数据过滤多路召回**：问题命中教材关键词时自动路由（规则表），向量路与 BM25 路各自带 `where` 过滤后再 RRF 融合、精排，跨教材候选去污；过滤后候选为空自动降级为全库检索（防过度过滤）；API / UI / MCP 三出口均可显式指定教材，显式指定失败报 400 带可用源清单而非静默降级
- 🖼️ **多模态语料接入（caption 路线）**：PyMuPDF bbox 渲染剪裁提取教材 PDF 内嵌图像（修复直接抽位图的反色/分层问题，自动合成矢量层；同页相邻对象合并 + 内容哈希去重），本地 qwen2.5-VL 批量生成中文描述，图像描述作为 image 类型 chunk 与文本**统一索引**（1434 块），命中图像块时答案引用区直接展示教材原图；图片类问题检索 top-8 命中率 8/8（全部 rank=1，n=3 校准集口径）
- 📌 **引用溯源（工程化）**：来源带相关性分数与原文偏移，可展开核对，也可直接喂评估；回答正文内联 `[n]` 上标（半角/全角均可），点击可跳转对应来源（P1-2）
- 💬 本地流式问答界面（Streamlit；快速模式**真实逐段流式输出**（ollama stream=True），精确模式因接地自检需完整答案而缓冲后整体渲染）
- 📊 **本地自实现 RAG 评估**：本机 Ollama 当裁判，四个核心指标测检索/回答质量，零额外依赖
- 🪵 **查询日志（P1-3）**：每轮对话后追加 `.query_log.jsonl`，记录模式 / 来源 / 接地判定 / 引用标记，便于事后回溯检索质量
- 🔌 **MCP 标准工具层（自实现协议子集）**：`mcp_server.py` 仅用 Python 标准库实现 MCP stdio 传输的最小协议子集（initialize 能力协商 / tools list / tools call，JSON-RPC 2.0），把混合检索暴露为标准 `course_retrieve` 工具（与 `/retrieve` 同一检索实现，fast/accurate 同语义）；经官方 MCP Inspector 互操作实测通过，任意 MCP 兼容 Agent Runtime 可直接挂载（证据见 `mcp_interop_evidence.md`）。零新依赖。
- ➕ **增量索引 + 去重（P1-4）**：稳定 id（`md5(source::start_index)`）自动去重，上传仅增量 upsert 新文件；按 source 删除单文档，其余不受影响，不再全量重建
- ✅ **核心算法单测（P2-1）**：零依赖 `test_core.py` 覆盖 BM25 排序 / RRF 融合去重 / 稳定 id 去重 / 接地判定 / 日志格式，脱离 chromadb/ollama/streamlit 亦可在任意干净 Python 环境实跑
- ✅ **集成测试 + CI（T9）**：`test_api.py`（30 用例，fastapi TestClient + mock 外部依赖，无需 Ollama 覆盖全部 API 端点/参数校验/路径穿越防护 + 统一闸门/流式聚合/鉴权/上传上限/引用图像字段单测）+ `test_retrieval.py`（11 用例，检索链路 + 教材过滤 + 图像块契约/真实索引 + 真实流式冒烟，无 Ollama 自动 skip）+ `test_mcp.py`（12 用例：11 个零依赖 MCP 协议/参数钳制用例 CI 直跑 + 1 个真实检索冒烟自动 skip）+ GitHub Actions（`.github/workflows/ci.yml`，3.11 矩阵，py_compile + 四套测试）
- 🔒 配置外置 `.env`，模型名/路径/调参不再写死代码

## 目录结构

```
CourseRAG/
├── config.py          # 所有参数集中从 .env 读取（配置外置；含 HYBRID_ENABLED / BM25_TOP_K）
├── rag_core.py        # RAG 核心：加载切分 / 嵌入建库 / 混合检索(hybrid_retrieve) + 重排 / 增量索引(add_documents/delete_by_source) / 拼链
├── rag_search_utils.py# 零依赖纯算法（P2-1 抽出）：分词 / BM25 / RRF 融合 / 稳定 id，供 test_core 独立测试
├── reranker.py        # bge-reranker 跨编码器重排（混合检索的精排段；默认开——难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败）
├── mcp_server.py      # MCP stdio 工具层：零依赖自实现协议子集（initialize/tools list/tools call），course_retrieve 与 /retrieve 同链路
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
# 检索模式（默认 accurate）：精确=向量 top-50 + BM25 融合（+ bge-reranker 精排）top-5（救难 query，A/B 实测 context_recall +16.7% / context_precision +19.2%）；快速=纯向量 top-5（降级档：GPU 不可用 / 显存不足 / 高并发时使用）。延迟因硬件而异——有 CUDA 时精排亚秒级，纯 CPU 会显著变慢
```

### MCP 工具层（把检索挂给任意 MCP 兼容 Agent）

`mcp_server.py` 是独立的 stdio 服务（与 Streamlit/API 并存，共用同一套检索实现，零新依赖）：

```powershell
# 手动验证（官方 Inspector，需 Node）：
npx -y @modelcontextprotocol/inspector --cli python mcp_server.py --method tools/list
npx -y @modelcontextprotocol/inspector --cli python mcp_server.py --method tools/call --tool-name course_retrieve --tool-arg query=什么是ACID --tool-arg mode=fast
# 任意 MCP 兼容客户端（如 Claude Desktop）按 stdio server 配置 command/args 即可挂载
```

工具 `course_retrieve`：入参 `query` / `top_k`(1-20，服务端强制钳制) / `mode`(fast|accurate)，返回带来源与相关性分数的文本块。互操作实测矩阵与协议细节见 `mcp_interop_evidence.md`。

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

> 部署层封装：把 `rag_core` 的「索引构建 / 检索 / 重排 / 生成」暴露为 RESTful 端点，
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

### 多用户预留扩展点（Gate 2）

> 决策：先本地/局域网部署让同学可用，**不重写后端**；以下扩展点默认关闭，需要多人公网暴露时再开。
> 实现：`api_security.py`（`require_auth` 全局依赖 + `RateLimitMiddleware` IP 滑动窗口），`main.py` 已挂载（no-op）。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `AUTH_ENABLED` | `false` | 开启后所有端点需 `X-API-Key` 头（值 = `API_KEY`） |
| `API_KEY` | 空 | 开启鉴权时设置的密钥 |
| `RATE_LIMIT` | 空 | 每 IP 每分钟请求上限（如 `60`），空 = 不限 |
| `CORS_ORIGINS` | `*` | 收紧跨域（生产填具体域名，如 `https://qa.example.com`） |
| `MAX_UPLOAD_MB` | `100` | `/documents` 上传大小上限（MB），超限返回 413；超大扫描版 PDF 走本地 `ingest_ocr.py` |

```bash
# 示例：开启鉴权 + 限流（docker compose 场景改环境变量或 .env）
AUTH_ENABLED=true API_KEY=your-secret RATE_LIMIT=60 python -m uvicorn main:app --port 8000
# 带密钥请求
curl -H "X-API-Key: your-secret" http://localhost:8000/health
```

### Docker 一键部署（双模式）

> **变更**：compose 改为 **API 直连宿主 Ollama**（删除了容器内 ollama 服务）。
> 原因：宿主已装 Ollama 且模型齐全，容器内再拉一套要重复下载 ~5GB 模型；
> 且避免 11434 端口冲突。**前置条件：宿主 Ollama 已启动且模型已拉取。**
> 实测验证（Windows Docker Desktop）：`host.docker.internal:11434` 直连宿主 Ollama 成功，
> /health、/query（fast + accurate）、/retrieve 全部正常，容器加载宿主挂载的索引（实测时 978 块，B 专项块去重 993→978；当前 1434 块，容器挂载宿主目录无需重建即可用新索引）。

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
- **已知坑（实测）**：多阶段构建只拷 site-packages，pip 装的可执行脚本不在
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

## 索引备份与语料接入（P0 / Gate 1）

**索引备份/恢复**（`backup_index.py`，纯文件操作，不依赖 Ollama）：

```bash
python backup_index.py backup   # 快照 chroma_db → chroma_db_backup_<时间戳>_<块数>/
python backup_index.py list     # 列出所有备份与块数
python backup_index.py restore  # 先备份当前、再恢复指定备份
```

**语料接入**（`ingest_ocr.py`，智能探测：有文字层 pypdf 直提，0 文字层降级 OCR）：

```bash
# 试点验证质量（先跑前 N 页，避免整本烧 1 小时+）
python ingest_ocr.py "data/教材\<书>.pdf" --max-pages 10
# 质量可接受 → 全量；无 --max-pages 时自动探测：直提 or OCR
python ingest_ocr.py "data/教材\<书>.pdf"
# 接入后重建索引（长任务，勿关终端；run_ab.py 内部强制 reranker off 仅用于 A/B 隔离）
python run_ab.py
```

> OCR 依赖（懒加载，未装不影响仓库）：`pip install rapidocr_onnxruntime pymupdf`
> （PyMuPDF 渲染 PDF 页 → RapidOCR/PP-OCRv4 识别中文，onnx 模型内置 wheel、离线可用。
> PaddleOCR 3.x 已判死勿装：paddlepaddle 3.3.1 Windows inference 加载 PIR JSON 模型必抛
> parse_error.101（最小复现钉死，见 ingest_ocr.py 头注释）。`data/教材` 原件一律保留不删）。

## 可观测看板 · 反馈闭环 · 语义缓存（P1/P2）

**可观测看板**（`dashboard.py`，零新依赖，读 `.query_log.jsonl`）：

```bash
python -m streamlit run dashboard.py   # 打开 http://localhost:8501
```

展示：KPI（累计/今日查询、平均延迟、拒答未接地率）、快速 vs 精确延迟均值 + P95 + 时序折线、
模式分布、来源 Top、拒答/未接地清单、👍/👎 反馈流。日志字段（`query_log.py` 写入）：

| 字段 | 含义 |
|---|---|
| `ts` | 毫秒级时间戳（反馈回写定位用） |
| `question` / `mode` | 问题与模式（快速/精确） |
| `top_sources` | 检索来源（rank/source/score） |
| `latency_ms` | 整轮问答耗时（P2 语义缓存命中时也如实记录） |
| `est_tokens` | 启发式估算 token（CJK≈1 token/字，非精确） |
| `rejected` / `grounded` | 硬拒答 / 接地判定 |
| `feedback` | 👍/👎（点击按钮后由 `set_feedback` 回写） |

**反馈闭环**（`feedback_pool.py`，把用户 👎 回流成评估集，不动冻结基线）：

```bash
python feedback_pool.py list                                   # 沉淀 👎 候选（去重）
python feedback_pool.py promote 0 --group B                    # 语料外应拒答 → B 组
python feedback_pool.py promote 1 --group C --expect reject    # 模糊 → C 组（需 --expect）
python eval_heldout.py --bc-only                               # 重跑门控覆盖回流用例
```

回流用例写入 `eval_data_feedback.json`，`eval_data_heldout.py` 运行时合并进 B/C 组——不直接改冻结常量，可审计可撤销。

**语义缓存**（`semantic_cache.py`，默认开，可关）：

```bash
# .env 可调：SEMANTIC_CACHE_ENABLED=true / SEMANTIC_CACHE_SIM=0.93 / SEMANTIC_CACHE_MAX=128
```

- 命中"语义相似"（embedding 余弦 ≥ 0.93）的重复/改写问句时，跳过检索 + reranker + 生成，直接返回历史答案，把精确模式整条链路（检索+精排 GPU 稳态约 1~3s + 生成段）压到 40ms 级。
- 只缓存正常答案（硬拒答/未接地不缓存）；**索引重建后自动失效**（`semantic_cache.invalidate()`）。
- 内存 LRU，单进程够用；多副本需换 Redis（见路线图路线 β/ε）。

**评估纪律（#18）**：`HELDOUT_A` 是已参与调参的**校准集**；对外正式宣布前用冻结测试集做最终回归：

```bash
python eval_test.py   # 跑 eval_data_test.TEST_SET（从未参与调参），结果入 eval_results_test.csv
```

当前官方基线与完整演化时间线（含生成提示温和化 → 切分修复 → 裁判口径修复 → 语料修复 → 裁判提取修复的 faithfulness 链路 0.811→0.781→0.897→0.976 单轮 → 0.886±0.023 旧裁判口径 → 0.912±0.021 → 0.901±0.022 当前）统一维护在「评估（本地自实现）→ 当前官方基线」一节，此处不再重复放数字。

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
- `context_precision`：召回结果里相关内容是否排在前面（reranker OFF/ON 与混合检索的实测增益见下方「当前官方基线」表）
- `context_recall`：检索是否覆盖了回答问题所需的信息（需 `eval_data.py` 里的 reference）

**对比用法（重点）**：先在 `RERANKER_ENABLED=false` 跑一次得【基线】，开 reranker 再跑一次，
两次 `context_precision` / `faithfulness` 的差值，就是 reranker 在你的数据上价值的硬证据——
这比"拍脑袋说 reranker 有用"有说服力得多。

评估样本在 `eval_data.py`（基于 `data/` 课程笔记 + `data/教材/` 真实教材的知识点，现 100 条），扩文档后往里追加即可。

### 当前官方基线（对外引用口径）

| 实验 | 结果 | 口径说明 |
|---|---|---|
| **冻结测试集**（24 题，从未参与调参；语料 1434 块（含 215 图像 caption 块）+ recall 明细重算口径） | faithfulness **0.901 ± 0.022** / answer_relevancy **0.975** / context_precision **0.869** / context_recall **0.948**（3 轮独立回归的统计值；检索侧两指标三轮逐位一致 std=0，方差全部来自生成侧 temp 0.2） | **对外引用的绝对值口径**；`python eval_test.py` 可复现 |
| **held-out A 组 v2**（38 题校准集 = 30 文本 + 8 图片题；门控 faith≥0.85 且 recall≥0.80） | faith **0.915** / relevancy 0.958 / precision 0.824 / recall **0.975** → **PASS**（余量充足） | 只准用它调参；A 组构成随图片题版本化（A v1 30 题存量数据不可比） |
| **图片命中率 top-8**（8 题图片问答，期望图像块进检索 top-8） | **8/8 = 100%，全部 rank=1 命中** | 期望块按 image_path 锚定；题面为自然用户问法 |
| **held-out B 组**（20 题语料外；门控拒答率≥90%） | **20/20 = 100% → PASS** | 硬拒答 + 接地自检 + 教材路由生效 |
| **混合检索 vs 纯向量**（隔离 reranker 的 A/B 差值） | precision **+0.18** / recall **+0.16** / faithfulness +0.09 / relevancy +0.05，四项全正向 | 100 题 × 697 块历史语料；**只看差值**（同裁判同语料） |
| **reranker OFF vs ON**（59 题全量语料） | precision +0.120 / relevancy +0.133 / recall +0.096 / faithfulness +0.032，四项全正向 | 8b 自评历史口径；reranker 救「正确块排 30+」的难 query（`RETRIEVE_K=50` 扩池 + 精排纠回） |
| **性能**（本地推理，数字硬件绑定跨机不可比） | fast 全链路 P50 **5.9s** / P95 13.4s（生成段主导）；accurate 检索+精排段 **GPU 稳态 P50 1.16s**（2026-09-03 GPU 化，CPU 基线 21.1s ≈ **18×** 提速）；语义缓存命中 6.4s→**40ms** 对冲 | 明细见本地 EVALUATION.md「性能基线」 |

> 注：context_recall 已按明细重算口径维护（自报口径 → 明细重算，漂移在裁判噪声带内）；历史单轮 faithfulness 0.976 经 3 轮独立回归证实为高水位采样；官方基线 0.886 ± 0.023 为旧裁判口径，裁判陈述提取修复后重测 3 轮为 0.912 ± 0.021（1219 块），教材过滤 + 多模态扩容（1434 块）后重测 3 轮为 0.901 ± 0.022（明细见本地 EVALUATION.md「裁判口径注记」）。

**裁判与噪声**：裁判为独立模型 qwen2.5:7b（与生成 llama3.1:8b 不同源，破自评偏差）；小模型判定噪声 ±0.02~0.03——结论只看方向与粗粒度差值，0.001 精度的绝对值无意义。已知边界：评估题与语料同源（自出题自答），recall/precision 可能虚高（详见本地 EVALUATION.md「已知限制」）。

### 基线演化时间线（faithfulness 主线；跨语料/跨裁判口径不可直接比较）

| 阶段 | 语料 / 裁判 | faithfulness | 说明 |
|---|---|---|---|
| 小样本起步（21 题） | 旧语料 / 8b 自评 | 0.577 | 暴露纯向量多主题下拉错文档的真问题 |
| 多主题扩展（33 题） | 5 篇笔记 / 8b 自评 | 0.686（reranker OFF） | 整体增益≈0，但难 query 正确块常排 30+ |
| 全量笔记（59 题） | 12 篇 / 8b 自评 | 0.639（reranker ON） | 四项全正向，reranker 价值清晰显现 |
| 真混合检索 v2（100 题） | 697 块 / qwen2.5:7b 独立裁判 | 0.81（混合）vs 0.72（纯向量） | BM25 全库独立召回，precision +0.18 / recall +0.16 |
| 冻结测试集立项（24 题） | 993→978 块 / 修复前裁判口径 | 0.811 → 0.781 | 生成提示温和化 +0.06；切分修复在裁判噪声带内 |
| 裁判口径修复 | 978 块 / 修复后口径 | 0.897 | 容错 JSON 提取 + 代码块整体判定 + 分数从明细重算；与旧口径不可比 |
| 语料修复（PyMuPDF） | **1219 块** / 修复后口径 | 0.976（单轮，后被证实为高水位采样） | 小程序 PDF 提取碎片化修复（列表渲染等题 0.5→1.0） |
| 多样本统计（旧裁判口径） | **1219 块** / 修复后口径 + recall 明细重算 | 0.886 ± 0.023（n=3） | 3 轮独立回归；检索侧指标（precision/recall）三轮零方差，方差全部来自生成侧 |
| 裁判陈述提取修复 | **1219 块** / 提取修复后口径 + recall 明细重算 | **0.912 ± 0.021（n=3）** | 修复裁判"多句只拆 1 条陈述"漏拆与概括句误判（Q15 三轮 0/0.5/0 → 0.91±0.1）；Q5 单轮 0.5 经 3 轮核实为噪声 |
| 教材过滤 + 多模态扩容（当前官方） | **1434 块**（+215 图像 caption 块）/ 同口径 | **0.901 ± 0.022（n=3）** | 教材级 filter 多路召回 + 图像统一索引：context_recall 0.872→**0.948**（多模态供给补强召回），context_precision 0.914→0.869（图块摊薄），faith 噪声带内持平；检索侧三轮零方差 |

历史逐题明细：`eval_compare_hybrid.md`（混合检索 A/B）与各 `eval_results_*.csv`——**历史评估文档与明细数据仅存本地，不入库**（仓库只含代码与 README）。

### held-out 门控评估（工程化 P0）

> 解决"评估集与语料同源 → 绝对值偏乐观"的问题：建**独立于训练集的 held-out 三组**，
> 用绝对指标 + 拒答率双门控，作为"能否称落地"的硬证据（详见工程化点评文档 0.0 总览）。

- 样本：`eval_data_heldout.py` = **A 组 v2 共 38**（语料内 30 + 图片题 8，答案依赖图中信息）+ **B 组 20**（语料外/应拒答）+ **C 组 10**（对抗/模糊/注入）。
- 门控：A 组 faithfulness ≥ **0.85** 且 context_recall ≥ **0.80**；B 组拒答率 ≥ **90%**。
- 运行（本机 Ollama，裁判 qwen2.5:7b 与生成 llama3.1:8b 不同源）：
  ```bash
  python eval_heldout.py --a-only    # A 组绝对指标（v1 口径 30 题）→ eval_results_heldout_A.csv
  python eval_heldout.py --a-v2      # A v2 口径（38 题含图片题）+ 图片命中率 → 独立 CSV
  python eval_heldout.py --bc-only   # B/C 拒答验证 → eval_results_heldout_BC.csv
  python eval_heldout.py             # 全量 + 总门控
  ```
- **实测**：A 组 / B 组门控结果见上方「当前官方基线」表（A v2 PASS 余量充足、B 组 100% PASS）。C 组无门控仅记录：期望应拒题全部正确拒（含提示注入），宽泛/辨析类问题按保守方向处理，Linux vs Windows 跨系统比较为既有误放通行为、多轮稳定。
- **图片命中率口径**：期望图像块（image_path 锚定）是否进检索 top-8；与 faith 评估同批题、两种口径分开算（faith 走全闸门链路，命中率只看检索层）。题面一律模拟真实用户自然问法（"教材的插图里…"这类元描述会引入词面噪声，实测压低命中率）。
- 调参依据：TOP_N 5→8（top-5 时正确块常排 6~8 名导致模型"无法完整回答"，faith 仅 0.746；调 8 后 +0.165，已固化 config.py）。
- ⚠️ **语料变更纪律**：每次语料/切分/检索参数变更后，必须重验 A 组与 B/C 门控，不能只看冻结测试集——语料扩增会改变检索重排（本轮 978→1219 块后：A 组校准集指标下移、B 组拒答 100%→95%，均为检索重排变化的直接体现；A 组单点波动含生成/裁判噪声，判断回归需看多轮而非单次）。
- ⚠️ 重跑 A 组前**先移走旧 CSV**（`eval_local.py` 有断点续跑，同名 CSV 会让全部题目被跳过）。
- 阈值可调：`config.py` 的 `HARD_REJECT_VECTOR_THRESHOLD` / `HARD_REJECT_RERANK_THRESHOLD` / `GROUNDED_SCORE_THRESHOLD`（改阈值重跑，**禁止为刷指标删 B 组题**）。

## 路线图（进行中）

- [x] **P1-1 混合检索**：自实现 BM25 + RRF 融合（向量 + 词面互补），零新依赖（config 新增 `HYBRID_ENABLED` / `BM25_TOP_K`）
- [x] **P1-2 内联引用**：回答正文 `[n]` 上标（半角/全角可点跳源），链复用同一份 docs 消除重复检索与来源不一致
- [x] **P1-3 查询日志**：每轮落 `.query_log.jsonl`（模式 / 来源 / 接地判定 / 引用标记）
- [x] **P1-4 增量索引 + 去重**：稳定 id 自动去重，上传仅增量 upsert，按 source 删单文档（不再全量重建）
- [x] **P1a 教材级元数据过滤多路召回**：关键词规则路由 + 双路带 filter + 空结果降级；A 组 faith 0.895→0.946 / recall 0.819→0.980，B 组拒答 19/20→20/20（校准集 v1 口径实测）
- [x] **P2 多模态（caption 路线）**：bbox 渲染剪裁提取 215 图 + qwen2.5-VL 批量中文描述 + 图文统一索引（1434 块）+ 引用区原图展示；图片命中率 top-8 8/8 全 rank=1；冻结测试集 recall 0.872→0.948（n=3 新基线）
- [x] **P2-1 核心算法单测**：零依赖 `test_core.py`，脱离 LLM 栈亦可实跑（7 用例全绿）
- [x] **T9 集成测试 + CI**：API 端点测试 29 用例（TestClient + mock，无需 Ollama）+ 检索冒烟 9 用例（无 Ollama 自动 skip）+ MCP 协议 12 用例 + GitHub Actions（3.11 矩阵，py_compile + 四套测试全绿）
- [x] 最小可跑闭环 + 配置外置 + 引用溯源
- [x] 接入 bge-reranker 重排并实测（混合检索；代码已验证可启用，难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败，现默认开）
- [x] 引用溯源工程化（分数 + 偏移，可喂评估）
- [x] 本地自实现 RAG 评估（小样本测检索/回答质量，本机 Ollama 当裁判，零额外依赖）
- [x] 中文 prompt + 中文感知切分（W4：prompt 改为中文防幻觉指令；切分器加中文分隔符。指标未明显提升，但暴露了跨主题检索拉错文档的真问题）
- [x] FastAPI 接口 + Docker：`main.py` / `api_schemas.py` / `Dockerfile` / `docker-compose.yml` / `requirements-api.txt`，见「API 服务」章节

## 说明

- **索引管理（P1-4）**：侧栏「🔄 全量重建索引（data/ 全部）」走 `get_vector_store(chunks)` 整体重建；上传文件走增量 upsert（稳定 id 自动去重）；删除单文档只按 source 从索引移除该文档，**其余文档不受影响**，不再全量重建。生产环境建议删除前「移入回收站 + 二次确认」。
- 混合检索默认开（`HYBRID_ENABLED=true`）：精确模式默认即为「向量 + BM25 融合」，即使 `RERANKER_ENABLED=false` 也非纯向量；reranker 已验证可启用：HF 镜像（hf-mirror.com）在本机可达，`bge-reranker-v2-m3` 权重已下载（约 2.2GB）。**默认 `RERANKER_ENABLED=true`**（难 query 正确块常排 30+，需 RETRIEVE_K=50 + reranker 纠回；33 样本整体增益≈0 但救难 query）。启动 app.py 须带 `HF_ENDPOINT=https://hf-mirror.com`（或离线），否则权重加载会去 huggingface.co 超时。
