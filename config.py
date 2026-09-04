"""
config.py — 所有可调参数集中外置到 .env，代码里不再写死模型名/路径。
这是工程规范的第一步：『配置与密钥分离』。
"""
import os
from dotenv import load_dotenv

# 读取项目根目录下的 .env（.env 不进 git，见 .gitignore）
load_dotenv()

# reranker 权重（bge-reranker-v2-m3）加载时需访问 HuggingFace；本机 huggingface.co 不可达会
# 超时卡死（表现为页面"一直转"）。故默认走 hf-mirror.com 镜像；可用 .env 的 HF_ENDPOINT
# 或 shell 环境变量覆盖（setdefault 不会覆盖已存在的同名变量）。
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

# ---------- 模型与运行时（来自 .env，给默认值便于本地直接跑）----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
# Ollama 默认本地端口；换机器/远程只需改这里
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------- RAG 调参（这些数为什么这么定）----------
# 注意：实际评估基线为 800/150（.env 与 README 一致；2026-08-22 统一口径，
# v2 A/B 等全部数字均在 CHUNK_SIZE=800 下测得）。勿再改回 500/80 以免与证据脱节。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # 每块字符数（教材正文段落约 200-800 字符）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # 块间重叠，避免切断句子/上下文
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # 低温度压制幻觉

# ---------- 重排 Reranker（召回多、精排少，两段式检索）----------
# 向量召回是双塔粗排（只看语义相似度），reranker 是单塔交互精排，
# 把 (query, doc) 拼一起过模型，捕捉细粒度相关性。生产 RAG 标配。
# 工程化落地（2026-08-24）：决策④ reranker 默认开启（生产 RAG 标配两段式）。
# 注意：run_ab.py 仍强制 RERANKER_ENABLED=false 隔离「BM25 纯贡献」A/B，不受此处影响；
# 快速模式（UI 默认）走 force_no_rerank，本机不触发 reranker，延迟不变；
# 仅「精确模式」叠加 bge-reranker 精排（延迟 19~115s，已在 UI 文案明示）。
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # 多语言，中文友好
# 运行设备（2026-09-03 GPU 化改造）：auto=有 CUDA 用 GPU，否则回退 CPU；可强制 cuda/cpu。
# 前提：venv 里的 torch 必须是 CUDA 轮子（pip install torch==2.13.0 --index-url
# https://download.pytorch.org/whl/cu126）；CPU 轮子装了也不会用 GPU。
# 注意 8GB 卡与 Ollama（llama3.1:8b 约 5GB）共存：pipeline 中精排与生成串行，
# 显存紧张时把 OLLAMA_KEEP_ALIVE 调短让两模型错峰。
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
# 半精度开关（默认关）：fp16 推理再快约一倍，但 rerank_score 会轻微漂移——
# 硬拒答阈值 HARD_REJECT_RERANK_THRESHOLD=0.30 按 fp32 校准，开 fp16 前必须重验 B 组拒答行为。
RERANKER_FP16 = os.getenv("RERANKER_FP16", "false").lower() == "true"
# GPU 上批量打分的 batch（16=候选池全量一次过；CPU 上该值无感）
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))       # 向量召回候选数（扩到 50：难 query 正确块常排 30+，池不够捞不到；配合 reranker 精排纠回）
# 重排后保留、进 LLM 的条数。2026-08-24 held-out A 组实测：5→8 后 faithfulness 0.746→0.911、
# context_recall 0.823→0.873（正确块常排 6~8 名，top-5 供给不足导致模型「无法完整回答」）。
# 不设更高（10+）：precision 已回落 0.811，噪声随 TOP_N 递增，0.85 门控余量已足。
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "8"))
# B 专项（2026-08-26）：RRF 融合候选池规模（>RERANKER_TOP_N）。
# 背景：Q21 软链接等难 query 的正确块只被 BM25 通道命中（向量通道缺席），
# 融合直接按 RERANKER_TOP_N=8 截断时其 RRF 分（仅 1/(rank+60)）被双通道块压制而淘汰，
# 永远进不了 rerank。扩到 16 让 BM25-only 正确块先入池，再由 reranker 精排回 top-8。
# 代价：rerank 候选翻倍 → 「精确模式」延迟上升（bge-reranker CPU 逐条打分）。
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "16"))

# ---------- 混合检索（向量 + 自实现 BM25，双通道 RRF 融合）----------
# 课程资料大量精确术语（ACID / 2NF / 死锁 / 事务隔离级别），纯向量对词面匹配不如 BM25。
# hybrid = 向量全库召回 top-K ‖ BM25 全库独立召回 top-K（两个通道互不依赖）→ RRF 倒数排名融合。
# BM25 自实现（零新依赖，更贴「自实现核心组件」的工程取舍说明）；融合分数落 metadata['hybrid_score']。
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "50"))  # BM25 通道独立召回数（对全库打分，不再复用向量候选池）

# ---------- P1a 元数据过滤多路召回（2026-09-04）----------
# 问题命中教材/课程关键词（rag_search_utils.SOURCE_RULES）时，向量路与 BM25 路都
# 限定到对应教材再 RRF 融合；未命中关键词 = 全库检索（行为与旧版一致）。
# 关闭后自动路由失效，但显式 source 参数（API/UI 指定教材）仍可用。
SOURCE_FILTER_ROUTING_ENABLED = os.getenv("SOURCE_FILTER_ROUTING_ENABLED", "true").lower() == "true"

# ---------- 路径 ----------
DATA_DIR = os.getenv("DATA_DIR", "./data")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

# ---------- 加载（T6: 扫描版 PDF 检测） ----------
# 均字符/页 低于此阈值视为无文字层（扫描版），建库时告警跳过而非静默空索引。
PDF_MIN_CHARS_PER_PAGE = int(os.getenv("PDF_MIN_CHARS_PER_PAGE", "30"))

# ---------- 评估裁判（T2: 独立裁判，2026-08-22） ----------
# 生成用 LLM_MODEL（llama3.1:8b）；裁判默认用独立模型，避免「同模型既生成又打分」的自评价偏差。
#   JUDGE_PROVIDER=local    → Ollama 本地独立模型（JUDGE_MODEL 指定，如 qwen2.5:7b，数据不出本机）
#   JUDGE_PROVIDER=deepseek → DeepSeek API（需 JUDGE_API_KEY；效果强，但数据出本机，谨慎用）
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "local")
# 2026-08-22 定稿：3b vs 7b 十题实测（judge_compare_3b7b.json）后选 7b——
# 拆分完整、判定符合语义、能识别检索失败场景；3b 会把无关内容判 True。
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", "https://api.deepseek.com/v1")

# ---------- 低置信阈值（T3: 拒答/提示机制，2026-08-22） ----------
# top-1 向量相似度（cosine）低于此值 → 判定「资料可能不足」，UI 显式提示而非硬答。
# 由 _thr_scan.py 实测：语料内 59 题 top1 p5=0.6896、语料外 10 题 max=0.7686（分布有重叠，
# nomic-embed-text 中文基线相似度偏高）→ 阈值取语料内 p5，误拒≈5%，漏拒部分语料外问题；
# 因此实现为「低置信提示」而非硬拒答。改阈值后重跑 _thr_scan.py 校准。
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.69"))

# ---------- 硬拒答 / 生成接地自检（P1，2026-08-24 工程化落地） ----------
# 软提示（LOW_CONFIDENCE_THRESHOLD）仍保留给「快速模式」；此处为「精确模式」的硬拒答闸门：
#   - 检索阶段：top-1 向量分 < HARD_REJECT_VECTOR_THRESHOLD 直接硬拒（不进生成）；
#     精确模式若开启 reranker，其最高 rerank_score < HARD_REJECT_RERANK_THRESHOLD 也判硬拒。
#   阈值取 0.55（低于软提示 0.69），仅拦明显语料外，误拒率更低。
#   - 生成阶段：GROUNDED_VERIFY_ENABLED 时，独立裁判判定回答是否完全基于上下文
#     （grounded），未接地则把答案替换为硬拒文案，杜绝软提示下的幻觉外溢。
# 实测校准：启用后跑 eval_heldout.py（B 组拒答率应 ≥90%），据结果回调阈值。
HARD_REJECT_VECTOR_THRESHOLD = float(os.getenv("HARD_REJECT_VECTOR_THRESHOLD", "0.55"))
HARD_REJECT_RERANK_THRESHOLD = float(os.getenv("HARD_REJECT_RERANK_THRESHOLD", "0.30"))
GROUNDED_VERIFY_ENABLED = os.getenv("GROUNDED_VERIFY_ENABLED", "true").lower() == "true"
# grounded 自检：回答被支撑比例（faithfulness 口径）低于此值 → 判未接地，触发硬拒替换。
GROUNDED_SCORE_THRESHOLD = float(os.getenv("GROUNDED_SCORE_THRESHOLD", "0.60"))

# ---------- 多用户预留扩展点（P2 / Gate 2，2026-08-24） ----------
# 默认全部关闭：本地 / 局域网部署零摩擦。将来要并发 Web 服务，开开关即可，不必重写后端。
#   - AUTH_ENABLED=true：所有 API 端点要求请求头 `X-API-Key: <API_KEY>`。
#   - RATE_LIMIT>0：按客户端 IP 限制每分钟请求数（内存令牌桶，单进程够用；
#     多副本需换 Redis，见文档路线 β/ε）。
#   - CORS_ORIGINS：允许源，生产收紧为具体域名（默认 "*" 仅演示 / 前后端分离）。
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "0"))            # 每 IP 每分钟请求上限，0=不限
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# 上传大小上限（MB，评审 P3，2026-08-30）：/documents 上传接口此前无大小限制，
# 任意大文件会被整体读入内存再写盘。100MB 覆盖正常课程资料（几百页文字层 PDF
# 通常 <20MB）；扫描版大 PDF 走本地 ingest_ocr.py 接入，不经上传接口。
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

# ---------- 语义缓存（P2，2026-08-25） ----------
# 缓解精确模式 reranker 延迟（精确段口径见 EVALUATION.md）：命中"语义相似"的重复/改写问句时跳过检索/重排/生成。
#   - SEMANTIC_CACHE_ENABLED：总开关（单用户可关；多人在线热问题重复率高时开着最划算）
#   - SEMANTIC_CACHE_SIM：embedding 余弦阈值，同义改写（0.93 上下）命中，意图不同不误命中
#   - SEMANTIC_CACHE_MAX：内存 LRU 上限（单进程；多副本需换 Redis，见文档路线 β/ε）
# 只缓存正常答案，硬拒答/未接地不缓存；索引重建后自动失效（semantic_cache.invalidate）。
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_SIM = float(os.getenv("SEMANTIC_CACHE_SIM", "0.93"))
SEMANTIC_CACHE_MAX = int(os.getenv("SEMANTIC_CACHE_MAX", "128"))

# ---------- 乱码清洗（2026-08-22：微信小程序 PDF CID 字体乱码） ----------
# 块级乱码字符占比 > 此阈值 → 整块丢弃（内容已不可读，纯噪声）；
# 否则剥离乱码字符保留正文。诊断：微信小程序 303 页 372 块中 29 块 >30%、
# 251 块 >10% 混入乱码（广泛存在）→ 阈值 0.3 只丢最脏块。
# 2026-08-22 修复：_is_good_char 不再整段放行拉丁补充区（0x00A0-0x00FF）——
# pypdf 对 CID 乱码的另一种映射是产出 æ/ç/è 等拉丁字符，原白名单误放行导致
# 698 块中 54 块残留；剔除后块数 698→697（1 块此前侥幸保留的乱码块被正确丢弃）。
GARBLED_DROP_THRESHOLD = float(os.getenv("GARBLED_DROP_THRESHOLD", "0.3"))
