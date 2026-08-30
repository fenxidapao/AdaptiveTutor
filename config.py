"""FeynmanTutor 配置：环境变量统一入口。

. 变量来源：项目根目录 .env（已从 deep-research 复制 DEEPSEEK_* 三行）
. 模型策略（PLAN 4.5）：默认 deepseek-v4-flash（稳定版），Ollama 7B 仅降级
"""

import os
from pathlib import Path

# 项目根目录（config.py 所在目录）
ROOT = Path(__file__).resolve().parent

# ---------- LLM（DeepSeek API 主力） ----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ---------- Ollama 降级（仅断网/欠费兜底，费曼追问环节强制禁用） ----------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# ---------- CourseRAG（知识底座，HTTP 调用） ----------
RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
# 检索模式：accurate=向量+BM25 融合（可选 reranker），带 hybrid/rerank 分数，讲解质量高（PLAN 4.6 默认）；
# fast=纯向量 top-N，秒级但返回 score 恒为 None（无法按相关度加权）。
# 注意：CourseRAG /retrieve 只认 question/mode，实际返回条数由其自身 RERANKER_TOP_N 决定，本端无法指定 top_k。
RAG_MODE = os.environ.get("RAG_MODE", "accurate")
# 客户端接收条数上限：作为截断保险，与 CourseRAG 返回条数取较小（实际以上游为准）
RAG_RETRIEVE_TOP_N = int(os.environ.get("RAG_RETRIEVE_TOP_N", "8"))
# 喂给 LLM 相关性过滤的单条文档内容截断预算（字符）
RAG_DOC_CHAR_LIMIT = int(os.environ.get("RAG_DOC_CHAR_LIMIT", "1200"))

# ---------- 状态库 ----------
DB_PATH = os.environ.get("DB_PATH", str(ROOT / "state.db"))

# ---------- 沙箱 ----------
SANDBOX_TIMEOUT = int(os.environ.get("SANDBOX_TIMEOUT", "5"))      # 秒
SANDBOX_MAX_OUTPUT = int(os.environ.get("SANDBOX_MAX_OUTPUT", "65536"))  # 字节

# ---------- C 阶段实验（PLAN 18.3） ----------
# EXPERIMENT_AUTH=1 时：业务端点强制 session 鉴权 + mode 按 group_name 强制 + 配额生效
# 默认 0（兼容本机演示/既有测试）；C 阶段部署时置 1
EXPERIMENT_AUTH = os.environ.get("EXPERIMENT_AUTH", "0") == "1"
# 每日每用户 LLM 调用上限（一次完整闭环约 12-15 次，重学多遍 50 次合理）
DAILY_LLM_LIMIT_PER_USER = int(os.environ.get("DAILY_LLM_LIMIT_PER_USER", "50"))
# 全站每日 LLM 调用总量上限（全局熔断，防单用户/异常刷爆 key）
GLOBAL_DAILY_LLM_LIMIT = int(os.environ.get("GLOBAL_DAILY_LLM_LIMIT", "300"))

# ---------- E 阶段 Loop 工程化（PLAN 20） ----------
# E1 学习回流：后测低于此阈值 且 提升不足 → 触发回流（重新学薄弱点后重测）
REFLOW_THRESHOLD = float(os.environ.get("REFLOW_THRESHOLD", "0.6"))
# 提升不足的定义：后测 - 前测 < 此值（起点极低但进步显著的不强留）
REFLOW_MIN_GAIN = float(os.environ.get("REFLOW_MIN_GAIN", "0.3"))
# 重测后测达标线（薄弱点是否补上的外部验证，非效果测量）
REFLOW_PASS_SCORE = float(os.environ.get("REFLOW_PASS_SCORE", "0.8"))
# 回流上限轮数（防无限循环烧钱/拖住用户）
REFLOW_MAX_ROUNDS = int(os.environ.get("REFLOW_MAX_ROUNDS", "2"))
# E3 每日学习队列上限（知识点数；配额是 LLM 次数，不宜直接当队列长度）
DAILY_QUEUE_LIMIT = int(os.environ.get("DAILY_QUEUE_LIMIT", "5"))

# ---------- ②Context 长对话压缩（9 步框架，PLAN 21 Pi 双层循环预留） ----------
# 费曼 transcript 超过该条数 → 触发压缩（3 轮=6 条不触发，双层循环轮数增长后生效）
CONTEXT_COMPRESS_AFTER = int(os.environ.get("CONTEXT_COMPRESS_AFTER", "8"))
# 压缩时保留原文的最近消息条数（追问需要最近的上下文）
CONTEXT_RAW_TAIL = int(os.environ.get("CONTEXT_RAW_TAIL", "4"))

# ---------- F 阶段 安全（五层纵深防御，PLAN 22） ----------
# L3 风险分级：管理操作（审计查看/用户列表等）仅限这些 user_id（EXPERIMENT_AUTH=1 时强制）
ADMIN_USER_IDS = [x.strip() for x in os.environ.get("ADMIN_USER_IDS", "u0").split(",") if x.strip()]
# L2 幂等：request_id → 首次响应的保留时长（秒），过期键顺带清理
IDEMPOTENCY_TTL_SECONDS = int(os.environ.get("IDEMPOTENCY_TTL_SECONDS", "3600"))
# L1 Prompt 边界：客户端 transcript 的条数上限与单条字符上限（LLM 边界硬截断）
TRANSCRIPT_MAX_MSGS = int(os.environ.get("TRANSCRIPT_MAX_MSGS", "40"))
TRANSCRIPT_MSG_CHARS = int(os.environ.get("TRANSCRIPT_MSG_CHARS", "2000"))


def load_dotenv(path: str | None = None) -> None:
    """极简 .env 加载：只读 KEY=VALUE 行，不覆盖已存在的环境变量。"""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# 模块加载时自动读 .env（保证 import config 即可用）
load_dotenv()
