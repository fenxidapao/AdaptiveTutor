# CourseRAG · 专业课/课本问答助手

基于 **本地 Ollama** 的垂直领域 RAG 应用，面向「计算机/数学专业课、课本知识问答」场景。
数据全程不出本机，不依赖任何云端 API。

> 技术栈：LangChain + Chroma + Ollama + Streamlit
> 定位：个人项目（AI 应用开发方向），在已跑通的 RAG-KnowBot 基础上做工程化改造与垂直落地。

---

## 功能

- 📚 多格式课程资料接入（PDF / TXT / MD）
- ✂️ 递归语义切分（带重叠窗口，避免切断上下文）
- 🔍 **两段式检索（reranker 默认开）**：向量召回 top-50（双塔粗排） + bge-reranker 精排 top-5（单塔交互细排；难 query 正确块常排 30+，靠扩大候选池 + reranker 纠正向量召回失败，见「评估基线」）
- 📌 **引用溯源（工程化）**：来源带相关性分数与原文偏移，可展开核对，也可直接喂评估
- 💬 本地流式问答界面（Streamlit）
- 📊 **本地自实现 RAG 评估**：本机 Ollama 当裁判，四个核心指标测检索/回答质量，零额外依赖
- 🔒 配置外置 `.env`，模型名/路径/调参不再写死代码

## 目录结构

```
CourseRAG/
├── config.py          # 所有参数集中从 .env 读取（配置外置）
├── rag_core.py        # RAG 核心：加载切分 / 嵌入建库 / 检索+重排 / 拼链
├── reranker.py        # bge-reranker 跨编码器重排（两段式检索的精排段；默认开——难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败）
├── app.py             # Streamlit 演示界面（含来源展示）
├── eval_local.py      # W3：本地自实现评估（Ollama 当裁判，4 指标，零额外依赖；取代已废弃的 eval.py）
├── eval_data.py       # 评估小样本（question + reference 标准答案）
├── data/              # 课程资料（示例：数据结构、数据库系统）
├── chroma_db/         # 向量库持久化目录（自动生成，已 gitignore）
├── .env.example       # 配置模板
└── requirements.txt
```

## 快速开始

```powershell
# 1. 准备环境（可复用你已有的 rag_env，或新建）
cd E:\WorkBuddy工作空间\2026-08-16-17-06-49\CourseRAG
python -m venv rag_env
.\rag_env\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置
cp .env.example .env

# 3. 启动 Ollama（已装则跳过 pull）
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 4. 运行界面
streamlit run app.py
# 浏览器打开 http://localhost:8501
# 侧栏确认 data/ 下示例 md 存在 → 点「索引 / 更新」→ 问「什么是 ACID？」应带来源回答
```

## 一键启动（本机自用，推荐）

> 本项目**唯一已验证可用的 Python 环境**是 `evalverify` venv（含 torch/torchvision/langchain/sentence-transformers 全家桶）。
> 仓库自带 `start.bat`，双击即用：

```powershell
# 方式一：双击 start.bat（已写好正确解释器路径）
# 方式二：手动
cd E:\WorkBuddy工作空间\2026-08-16-17-06-49\CourseRAG
C:\Users\user\.workbuddy\binaries\python\envs\evalverify\Scripts\python.exe -m streamlit run app.py
```

⚠️ 不要用 `CourseRAG/rag_env`（仅 9 个包的空壳）或别的项目的 venv（缺 torchvision），否则 `ModuleNotFoundError`。

### 干净部署（换机器 / 从头装）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # langchain 全家桶已精确锁定；torch/transformers 有上下限
ollama pull nomic-embed-text
ollama pull llama3.1:8b
streamlit run app.py
```

`requirements.txt` 的版本锁是关键：langchain 全家桶必须同线，混装会触发 `is_data_content_block` 等符号冲突。

## 架构（离线构建 / 在线问答）

```
数据接入 → 切分(800/150) → 嵌入(nomic-embed-text) → Chroma 持久化
                                                    ↓
问题 → 嵌入 → 向量召回 top-10（双塔粗排）
                  ↓
            [bge-reranker 精排 top-5]（单塔交互；现已默认开——难 query 正确块常排 30+，靠 RETRIEVE_K=50 + reranker 纠回）
                  ↓
            拼 Prompt → llama3.1:8b 生成 → 回答 + 来源引用（带分数/偏移）
```

## 相对 RAG-KnowBot 的改造

| 项 | RAG-KnowBot | CourseRAG |
|---|---|---|
| 配置 | 硬编码在文件顶部 | 全部外置 `.env` |
| 模块化 | 单文件 | `config` / `rag_core` / `reranker` 分离 |
| 检索 | 单阶段向量召回 top-5 | 两段式：向量召回 top-10 + reranker 精排 top-5（可选） |
| 引用 | 折叠面板贴片段 | 结构化：排名 + 相关性分数 + 原文偏移 |
| 评估 | 无 | 本地自实现评估（Ollama 当裁判：faithfulness / answer_relevancy / context_precision / context_recall） |
| 领域 | 通用 demo | 垂直：专业课/课本问答 |
| 模型端点 | 默认本地 | 显式 `base_url`，便于换远程/容器 |

## 评估（本地自实现，W3）

> ⚠️ 关于 RAGAS：曾尝试用 RAGAS，但它 0.1.22 / 0.2.x 在 import 时会拉取
> `langchain_community.chat_models.vertexai`，而 langchain_community 0.3.x 已移除该符号，
> 无法与本项目 langchain 0.3 全家桶共存（pip 还会把 0.1.22 自动升回 0.2.x）。
> 故**放弃 RAGAS**，改成本机 Ollama 当裁判，**自己实现**四个指标——零额外依赖，反而更贴合
> "AI 应用开发·自己实现关键组件"的工程取舍说明。

纯本地评估，**不碰 HuggingFace**：裁判 LLM 用本机 Ollama（temperature=0 逼稳定输出 JSON），
embedding 用 nomic-embed-text。

```powershell
# 评估脚本只复用 langchain_ollama，无需 ragas/datasets/nest_asyncio
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

评估样本在 `eval_data.py`（基于 `data/` 两篇笔记的真实知识点），扩文档后往里追加即可。

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

> **扩展语料说明（已全量评测）**：`data/` 已扩至 **12 篇**（新增排序算法 / 图与最短路径 / 关系代数与 SQL / 函数依赖与范式 / OS 内存管理 / 计网 IP 与子网路由 / 组成原理指令系统与寻址），`eval_data.py` 扩至 **59 条**，`config.py` 已调 `CHUNK_SIZE=500 / CHUNK_OVERLAP=80 / RETRIEVE_K=50`（候选池扩到 50，让排 30+ 的难 query 正确块也能进池，reranker 才能纠回）。该扩展的 OFF/ON 全量对比**已于 2026-08-18 执行完毕**（见上方「59 样本评估基线」），**最终结论以 59 样本全量为准**；上方 33 样本表作为子集参考保留。重跑命令见交接文档第 4 节（ON 轮须带 `HF_ENDPOINT=https://hf-mirror.com`）。

## 路线图（进行中）

- [x] 最小可跑闭环 + 配置外置 + 引用溯源
- [x] 接入 bge-reranker 重排并实测（两段式检索；代码已验证可启用，难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败，现默认开）
- [x] 引用溯源工程化（分数 + 偏移，可喂评估）
- [x] 本地自实现 RAG 评估（小样本测检索/回答质量，本机 Ollama 当裁判，零额外依赖）
- [x] 中文 prompt + 中文感知切分（W4：prompt 改为中文防幻觉指令；切分器加中文分隔符。指标未明显提升，但暴露了跨主题检索拉错文档的真问题）
- [ ] （可选）FastAPI 接口 + Docker

## 说明

- 删除 `data/` 中文件后会用剩余文档整体重建索引；生产环境建议改为「移入回收站 + 二次确认」。
- reranker 已验证可启用：HF 镜像（hf-mirror.com）在本机可达，`bge-reranker-v2-m3` 权重已下载（约 2.2GB）。**默认 `RERANKER_ENABLED=true`**（难 query 正确块常排 30+，需 RETRIEVE_K=50 + reranker 纠回；33 样本整体增益≈0 但救难 query）。启动 app.py 须带 `HF_ENDPOINT=https://hf-mirror.com`（或离线），否则权重加载会去 huggingface.co 超时。
