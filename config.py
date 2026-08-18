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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # 每块字符数（扩语料实验中调细，给 reranker 更多候选）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # 块间重叠，避免切断句子/上下文
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # 低温度压制幻觉

# ---------- 重排 Reranker（召回多、精排少，两段式检索）----------
# 向量召回是双塔粗排（只看语义相似度），reranker 是单塔交互精排，
# 把 (query, doc) 拼一起过模型，捕捉细粒度相关性。生产 RAG 标配。
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # 多语言，中文友好
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))       # 向量召回候选数（扩到 50：难 query 正确块常排 30+，池不够捞不到；配合 reranker 精排纠回）
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))  # 重排后保留、进 LLM 的条数

# ---------- 路径 ----------
DATA_DIR = os.getenv("DATA_DIR", "./data")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
