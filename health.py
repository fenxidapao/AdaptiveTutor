"""
health.py — 启动自检（P0-1）。

把 check_env.py 的核心检查抽成纯函数，供 app.py 在侧边栏顶部渲染状态条，
让用户一打开就知道环境是否就绪，而不是点提问后才对着 spinner 干瞪眼。

本文件不依赖 streamlit，保持纯逻辑，便于复用与测试。
"""
import os
import json
import urllib.request

import config


def check_ollama(base_url=None, timeout=4):
    """检查本地 Ollama 与生成链路是否就绪。返回 (ok: bool, msg: str)。

    C3（2026-10-15）起生成可切云端；provider=openai 时 Ollama 只负责嵌入，
    不再要求本地存在 LLM_MODEL，改为检查云端 key 是否可解析（只报有无，不回显）。
    """
    cloud = False
    judge_cloud = False
    try:
        import llm_provider

        cloud = llm_provider.provider_name() == "openai"
        judge_cloud = llm_provider.judge_provider_name() == "openai"
    except Exception:
        cloud = False
        judge_cloud = False

    base = base_url or config.OLLAMA_BASE_URL
    url = base.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            tags = json.load(resp).get("models", [])
        names = {m.get("name") for m in tags}

        def _has(name):
            # Ollama 常带 :latest 标签；llama3.1:8b 的冒号是标签本身
            return name in names or any(n.startswith(name + ":") for n in names)

        need = {config.EMBEDDING_MODEL}
        if not cloud:
            need.add(config.LLM_MODEL)
        if not judge_cloud:
            need.add(config.JUDGE_MODEL)
        lack = {x for x in need if not _has(x)}
        if lack:
            return False, "Ollama 已启动，但缺模型：%s（运行 ollama pull %s）" % (
                ", ".join(sorted(lack)),
                " ".join(sorted(lack)),
            )
        if cloud or judge_cloud:
            if not llm_provider.key_available():
                return False, "生成/裁判已切云端，但未找到 API key（LLM_API_KEY / KUAIPAO_API_KEY / ~/.dsh/.credentials.yaml）"
            parts = []
            if cloud:
                parts.append("生成=%s" % config.LLM_MODEL)
            else:
                parts.append("生成=本地 %s" % config.LLM_MODEL)
            if judge_cloud:
                parts.append("裁判=%s" % config.JUDGE_MODEL)
            else:
                parts.append("裁判=本地 %s" % config.JUDGE_MODEL)
            return True, "Ollama 运行中（嵌入本地），" + "，".join(parts)
        return True, "Ollama 运行中，嵌入/生成模型齐全"
    except Exception:
        return False, "Ollama 未启动或不可达（%s）→ 终端运行 ollama serve" % url


def check_reranker_cache():
    """reranker 权重是否已缓存（决定首次启用的联网下载）。返回 (ok: bool, msg: str)。"""
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
    )
    if os.path.isdir(cache):
        return True, "reranker 权重已缓存（可离线加载，不依赖外网）"
    return False, "reranker 权重未缓存：首次启用会联网下载 ~2.2GB（app.py 已默认走镜像）"


_DEFAULT_DEPS = [
    "ollama",
    "chromadb",
    "streamlit",
    "sentence_transformers",
    "pypdf",
    "docx",
]


def check_deps(required=None):
    """关键依赖是否可导入。返回 (ok: bool, msg: str)。
    required 可传入自定义依赖清单（如 API 服务只需核心栈，无需 streamlit/torch）。"""
    required = required or _DEFAULT_DEPS
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return False, "缺依赖：%s（pip install -r requirements.txt）" % ", ".join(missing)
    return True, "依赖齐全"


def collect_status(deps=None):
    """汇总所有检查，返回 [(label, ok, msg), ...]，供 UI / API 渲染。"""
    return [
        ("依赖", *check_deps(deps)),
        ("Ollama", *check_ollama()),
        ("Reranker 权重", *check_reranker_cache()),
    ]
