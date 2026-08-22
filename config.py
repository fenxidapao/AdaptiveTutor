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
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # 多语言，中文友好
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))       # 向量召回候选数（扩到 50：难 query 正确块常排 30+，池不够捞不到；配合 reranker 精排纠回）
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))  # 重排后保留、进 LLM 的条数

# ---------- 混合检索（向量 + 自实现 BM25，双通道 RRF 融合）----------
# 课程资料大量精确术语（ACID / 2NF / 死锁 / 事务隔离级别），纯向量对词面匹配不如 BM25。
# hybrid = 向量全库召回 top-K ‖ BM25 全库独立召回 top-K（两个通道互不依赖）→ RRF 倒数排名融合。
# BM25 自实现（零新依赖，更贴「自实现核心组件」的工程取舍说明）；融合分数落 metadata['hybrid_score']。
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "50"))  # BM25 通道独立召回数（对全库打分，不再复用向量候选池）

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
