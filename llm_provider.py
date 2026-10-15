"""llm_provider.py — 生成模型统一提供方（本地 Ollama / OpenAI 兼容云端）。

背景（C3 生成切云，2026-10-15）
--------------------------------
CourseRAG 原本「生成」走本地 Ollama（``llama3.1:8b``，实测约 4.9 GB 显存）；
「裁判」走本地 Ollama（``qwen2.5:7b``，实测约 4.7 GB）。两者同机部署超过本机
8 GB 物理显存，Windows WDDM 不会报 OOM，而是把 CUDA 调用挂住（日志长时间无写入、
CPU 仍上涨），极易被误判成死锁。故生成先切到云端 OpenAI 兼容接口。

设计约束
--------
1. **模型名一律来自 config.LLM_MODEL**，代码里不硬编码模型名。
2. **API key 只从环境变量或 ``~/.dsh/.credentials.yaml`` 读取**，绝不写入仓库、
   绝不回显到日志、异常信息或响应体。
3. **默认 provider=ollama**：不设置 ``LLM_PROVIDER`` 时行为与旧版完全一致，
   保证可以随时回滚。
4. **预算自适应**：``deepseek-v4.1-flash`` 是推理模型，实测 ``max_tokens=256``
   时 256 个 token 全部耗在 reasoning，``content`` 为空且 ``finish_reason=length``。
   因此空 content + reasoning 占满预算时自动翻倍重试，直到 ``LLM_MAX_TOKENS_CAP``。

对外接口
--------
``resolve_api_key()`` / ``chat(messages, temperature=None, max_tokens=None)`` /
``chat_stream(messages, temperature=None, max_tokens=None)``。
"""
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Iterator

import config


class LLMError(RuntimeError):
    """一次性 LLM 调用失败（网络/鉴权/预算耗尽/无有效输出）。"""


class ThinkingUnsupported(LLMError):
    """该模型不认关闭思维链的字段（上游 400/422）。调用方应换一个变体重试。"""


# --------------------------------------------------------------------------
# 凭据解析（绝不回显 key 本身）
# --------------------------------------------------------------------------
_CRED_FILE = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
_CRED_PATTERN = re.compile(r"KUAIPAO_API_KEY\s*:\s*(\S+)")


def resolve_api_key() -> str:
    """按优先级取中转 API key：显式配置 > 环境变量 > 凭据文件。

    返回值只用于组装 Authorization 头，调用方不得写日志、不得入库。
    """
    for name in ("LLM_API_KEY", "KUAIPAO_API_KEY", "VISUAL_RERANK_API_KEY"):
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    try:
        text = _CRED_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _CRED_PATTERN.search(text)
    return match.group(1) if match else ""


def key_available() -> bool:
    """供 health 检查使用：只报告有无，不泄漏内容。"""
    return bool(resolve_api_key())


# --------------------------------------------------------------------------
# 本地 Ollama
# --------------------------------------------------------------------------
_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        _ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
    return _ollama_client


def _ollama_chat(messages, temperature):
    opts = {"temperature": temperature} if temperature is not None else None
    resp = _get_ollama_client().chat(
        model=config.LLM_MODEL, messages=messages, options=opts
    )
    return resp["message"]["content"]


def _ollama_chat_stream(messages, temperature):
    opts = {"temperature": temperature} if temperature is not None else None
    stream = _get_ollama_client().chat(
        model=config.LLM_MODEL, messages=messages, options=opts, stream=True
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield piece


# --------------------------------------------------------------------------
# OpenAI 兼容云端
# --------------------------------------------------------------------------
def _redact(text: str) -> str:
    """把错误文案里可能出现的 API key 替换掉。

    为什么必须做：上游（中转/网关）在 401/403 时经常把请求头原样回显到
    错误体里。若我们原样抛出，key 就会进入日志、API 响应和测试输出。
    这里对所有已知来源的 key 做子串替换，宁可少一点排查信息，也不泄漏凭据。
    """
    if not text:
        return text
    for name in ("LLM_API_KEY", "KUAIPAO_API_KEY", "VISUAL_RERANK_API_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    try:
        file_key = resolve_api_key()
    except Exception:
        file_key = ""
    if file_key:
        text = text.replace(file_key, "***")
    return text


def _cloud_base_url() -> str:
    base = config.LLM_API_BASE or config.OLLAMA_BASE_URL
    return base.rstrip("/")


def _cloud_headers() -> dict:
    key = resolve_api_key()
    if not key:
        raise LLMError(
            "云端生成缺少 API key：请设置 LLM_API_KEY / KUAIPAO_API_KEY 环境变量，"
            "或写入 ~/.dsh/.credentials.yaml（键名 KUAIPAO_API_KEY）。"
        )
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }


def _parse_cloud_response(body: dict) -> tuple[str, str, str]:
    """从 /chat/completions 响应里取 (content, reasoning, finish_reason)。

    content 可能是字符串或分段数组，统一拼成字符串。
    """
    choices = body.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    reasoning = message.get("reasoning_content") or ""
    return (content or "").strip(), (reasoning or "").strip(), choice.get("finish_reason") or ""


# 关闭思维链的候选参数：不同中转/模型群认不同字段，逐个试到成功。
# 实测（2026-10-15，真实 RAG 提示）：deepseek-v4.1-flash 支持
# reasoning_effort=none / thinking.type=disabled / enable_thinking=false。
_THINKING_OFF_ATTEMPTS = (
    {"reasoning_effort": "none"},
    {"thinking": {"type": "disabled"}},
    {"enable_thinking": False},
)


def _cloud_post(messages, temperature, max_tokens, thinking_off: bool = False,
                thinking_style: int = 0) -> dict:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0.2 if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    if thinking_off and 0 <= thinking_style < len(_THINKING_OFF_ATTEMPTS):
        payload.update(_THINKING_OFF_ATTEMPTS[thinking_style])
    req = urllib.request.Request(
        _cloud_base_url() + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_cloud_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _redact(exc.read().decode("utf-8", "replace")[:300])
        except Exception:
            pass
        # 上游不认 reasoning_effort / thinking / enable_thinking 时通常回 400/422；
        # 交给调用方换下一个变体，而不是整轮判死。
        if thinking_off and exc.code in (400, 422):
            raise ThinkingUnsupported(
                "HTTP %s 不接受该关闭思考参数：%s" % (exc.code, detail)
            ) from exc
        raise LLMError(
            "云端返回 HTTP %s（模型 %s）：%s" % (exc.code, config.LLM_MODEL, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMError("云端网络不可达（%s）：%s" % (_cloud_base_url(), exc.reason)) from exc
    except json.JSONDecodeError as exc:
        raise LLMError("云端返回非 JSON 响应：%s" % exc) from exc


def _cloud_chat(messages, temperature, max_tokens=None) -> str:
    """云端非流式对话：关闭思维链 → 换参数 → 预算翻倍，三重兜底。

    为什么这么复杂（2026-10-15 实测，真实 RAG 提示 1868 字符 / 8 块上下文）：
      - deepseek-v4.1-flash 默认开启思维链，3/3 次把 max_tokens 全烧在
        reasoning 上（7000~10000 字符），content 为空且 finish=length；
      - 传 reasoning_effort=none 或 thinking.type=disabled 后 0 reasoning、
        正常产出 content。但不同模型认的字段不同，故逐个试；
      - 即使关了思考，仍保留预算翻倍，避免别的推理模型重演同一问题。
    """
    budget = max_tokens or config.LLM_MAX_TOKENS
    cap = max(config.LLM_MAX_TOKENS_CAP, budget)
    last_reason = ""
    if config.LLM_DISABLE_THINKING:
        styles = range(len(_THINKING_OFF_ATTEMPTS))
    else:
        styles = range(1)
    for style in styles:
        b = budget
        while True:
            try:
                body = _cloud_post(
                    messages, temperature, b,
                    thinking_off=config.LLM_DISABLE_THINKING, thinking_style=style,
                )
            except ThinkingUnsupported as exc:
                # 该模型不认这个关闭思考的字段：换下一个变体
                last_reason = str(exc)
                break
            content, reasoning, finish = _parse_cloud_response(body)
            if content:
                return content
            last_reason = "finish=%s reasoning_chars=%d budget=%d style=%d" % (
                finish, len(reasoning), b, style,
            )
            if reasoning and finish == "length" and b < cap:
                b = min(b * 2, cap)
                continue
            break
    raise LLMError(
        "云端无有效输出（%s）。该模型可能持续把额度耗在推理上；"
        "可切换 LLM_MODEL（实测 kimi-k2.6 在真实 RAG 提示下 3/3 稳定）"
        "或调 LLM_MAX_TOKENS / LLM_MAX_TOKENS_CAP。" % last_reason
    )

def _cloud_chat_stream(messages, temperature, max_tokens=None) -> Iterator[str]:
    budget = max_tokens or config.LLM_MAX_TOKENS
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0.2 if temperature is None else temperature,
        "max_tokens": budget,
        "stream": True,
    }
    if config.LLM_DISABLE_THINKING:
        # 与 _cloud_chat 同因：推理模型可能把预算全烧在 reasoning 上，
        # 流式下表现为长时间没有 delta.content。默认关闭思考。
        payload.update(_THINKING_OFF_ATTEMPTS[0])
    req = urllib.request.Request(
        _cloud_base_url() + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_cloud_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0] or {}
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield piece
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _redact(exc.read().decode("utf-8", "replace")[:300])
        except Exception:
            pass
        raise LLMError(
            "云端流式返回 HTTP %s（模型 %s）：%s" % (exc.code, config.LLM_MODEL, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMError("云端流式网络不可达（%s）：%s" % (_cloud_base_url(), exc.reason)) from exc


# --------------------------------------------------------------------------
# 对外统一入口
# --------------------------------------------------------------------------
def provider_name() -> str:
    value = (config.LLM_PROVIDER or "ollama").strip().lower()
    return value if value in ("ollama", "openai") else "ollama"


def chat(messages, temperature=None, max_tokens=None) -> str:
    """非流式对话，返回纯文本。失败抛 LLMError。"""
    if provider_name() == "openai":
        return _cloud_chat(messages, temperature, max_tokens)
    return _ollama_chat(messages, temperature)


def chat_stream(messages, temperature=None, max_tokens=None) -> Iterator[str]:
    """流式对话，逐片段 yield 文本。失败抛 LLMError。"""
    if provider_name() == "openai":
        yield from _cloud_chat_stream(messages, temperature, max_tokens)
        return
    yield from _ollama_chat_stream(messages, temperature)
