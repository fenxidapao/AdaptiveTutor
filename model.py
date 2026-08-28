"""LLM 封装：DeepSeek API 调用（抄 deep-research model.py 思路，减薄为纯 HTTP）。

. 思路照抄：重试 3 次 + 空输出兜底 + 402 余额预警
. 实现减薄：标准库 urllib 直连 OpenAI 兼容 /chat/completions，零第三方依赖
. 费曼追问环节必须走 API（7B 会盲目附和讲错内容），见 feynman.py 调用约束

用法：
    from model import chat, chat_with_fallback
    text = chat([{"role": "user", "content": "你好"}])
"""

import contextvars
import json
import time
import urllib.error
import urllib.request

import config

MAX_RETRIES = 3
TIMEOUT = 60  # 单次请求超时（秒）
MAX_BUDGET = 8000  # 推理预算扩容上限（max_tokens 不会超过它）

# 可观测性 hook：外部用 set_log_hook 注册（见 db.log_llm_call），
# 每次成功的 LLM 调用会回调 (caller, model, usage, latency_ms, attempts, expanded, source)。
LOG_HOOK = None

# 当前请求用户（Web 层设置，hook 落库带 user_id 供配额计费，PLAN 18.3）
_CURRENT_USER = contextvars.ContextVar("ft_current_user", default=None)
# 当前请求会话（D4 session trace：一次闭环可回放，PLAN 18.5）
_CURRENT_SESSION = contextvars.ContextVar("ft_current_session", default=None)


def set_current_user(user_id: str | None) -> None:
    """设置本次请求的发起用户（contextvar，线程安全）。None 清除。"""
    _CURRENT_USER.set(user_id)


def get_current_user() -> str | None:
    return _CURRENT_USER.get()


def set_current_session(session_id: str | None) -> None:
    """设置本次请求的会话（session trace 用）。None 清除。"""
    _CURRENT_SESSION.set(session_id)


def get_current_session() -> str | None:
    return _CURRENT_SESSION.get()


def set_log_hook(fn) -> None:
    """注册 LLM 调用日志回调（PLAN 8 多 Agent 效率评估）。fn=None 关闭。"""
    global LOG_HOOK
    LOG_HOOK = fn


class ModelError(Exception):
    """LLM 调用失败（网络/401/429/402/空输出等）。调用方决定是否降级。"""


class ReasoningBudgetError(ModelError):
    """推理模型把 max_tokens 预算用在了思维链（reasoning_content）上，content 为空。

    deepseek-v4-flash 是推理模型：max_tokens 是"推理+回答"的总预算。
    遇到此类错误应自动扩容重试（见 chat 内 MAX_BUDGET_SCALE）。
    """

    def __init__(self, message: str, max_tokens: int):
        super().__init__(message)
        self.max_tokens = max_tokens


def _post_chat(messages: list[dict], temperature: float, max_tokens: int,
               base_url: str, api_key: str, model: str) -> tuple[str, dict, int]:
    """调用一次 /chat/completions，返回 (纯文本, usage, 耗时毫秒)。异常由上层处理。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = int((time.monotonic() - t0) * 1000)
    choice = body["choices"][0]["message"]
    content = choice.get("content", "")
    if isinstance(content, list):  # 某些模型的 content 是分段数组
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    text = (content or "").strip()
    if not text:
        # deepseek-v4-flash 是推理模型：先输出 reasoning_content 再输出 content。
        # 若 max_tokens 太小，推理阶段就把预算用光，content 为空。
        reasoning = choice.get("reasoning_content")
        if reasoning:
            raise ReasoningBudgetError(
                f"模型只输出了推理内容（max_tokens={max_tokens} 可能太小，"
                f"推理阶段已用尽预算），未产出正式回答",
                max_tokens=max_tokens)
    usage = body.get("usage") or {}
    return text, usage, latency_ms


def _warn_balance(http_code: int, body_text: str) -> None:
    """402 = 余额不足，明确预警（deep-research 已验证的坑）。"""
    if http_code == 402:
        print("[!] 警告：DeepSeek API 返回 402，余额可能不足，请充值后重试。")
    elif http_code in (401, 403):
        print(f"[!] 警告：API 鉴权失败（HTTP {http_code}），请检查 DEEPSEEK_API_KEY。")
    else:
        print(f"[!] LLM 请求失败：HTTP {http_code} -> {body_text[:200]}")


def _request_once(messages, temperature, max_tokens, *, allow_local: bool):
    """先试 API；allow_local 时失败降级 Ollama。单次尝试（不做重试）。

    返回 (text, usage, latency_ms, source)。usage 可能为空 dict（mock/异常路径）。
    """
    # 1) 主力：DeepSeek API
    try:
        text, usage, latency = _post_chat(
            messages, temperature, max_tokens,
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
            model=config.DEEPSEEK_MODEL,
        )
        return text, usage, latency, "api"
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        _warn_balance(e.code, body_text)
    except urllib.error.URLError as e:
        print(f"[!] 网络错误：{e.reason}")
    except ReasoningBudgetError:
        raise  # 让 chat() 捕获并扩容重试
    except Exception as e:  # 解析失败/超时等
        print(f"[!] API 调用异常：{type(e).__name__}: {e}")

    # 2) 降级：Ollama 7B（仅 allow_local=True）
    if allow_local:
        try:
            print("[i] 降级到本地 Ollama qwen2.5:7b ...")
            text, usage, latency = _post_chat(
                messages, temperature, max_tokens,
                base_url=config.OLLAMA_BASE_URL + "/v1",
                api_key="ollama",
                model=config.OLLAMA_MODEL,
            )
            return text, usage, latency, "ollama"
        except Exception as e:
            print(f"[!] Ollama 降级也失败：{e}")
    return "", {}, 0, "api"


def _budget_retry_once(messages, temperature, max_tokens, *, allow_local):
    """带推理预算扩容的单次尝试：预算不足时 max_tokens 翻倍重试（最多扩到 MAX_BUDGET）。

    返回 (text, expanded, usage, latency_ms, source)。非预算错误不在这里处理（交给上层重试）。
    """
    current = max_tokens
    while True:
        try:
            text, usage, latency, source = _request_once(
                messages, temperature, current, allow_local=allow_local)
            return text, (current != max_tokens), usage, latency, source
        except ReasoningBudgetError as e:
            if current >= MAX_BUDGET:
                print(f"[!] 推理预算已达上限 {MAX_BUDGET}，仍无正式回答")
                return "", True, {}, 0, "api"
            current = min(current * 2, MAX_BUDGET)
            print(f"[i] 推理预算不足（max_tokens={e.max_tokens}），扩容到 {current} 重试...")


def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2000,
         caller: str = "") -> str:
    """DeepSeek API：重试 3 次 + 推理预算自动扩容 + 空输出兜底 + 402 预警。

    失败（重试耗尽或空输出）抛 ModelError，调用方决定降级或中止。
    用于诊断/讲解/测评等允许降级的环节。caller 标记调用环节（可观测性）。
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text, expanded, usage, latency, source = _budget_retry_once(
                messages, temperature, max_tokens, allow_local=False)
        except Exception as e:
            last_err = e
            text = ""
            expanded, usage, latency, source = False, {}, 0, "api"
        if text:
            _fire_hook(caller, usage, latency, attempt, expanded, source)
            return text
        if attempt < MAX_RETRIES:
            time.sleep(1.5 * attempt)
    raise ModelError(f"LLM 调用失败（重试 {MAX_RETRIES} 次后无有效输出）：{last_err}")


def chat_with_fallback(messages: list[dict], temperature: float = 0.3,
                       max_tokens: int = 2000, allow_local: bool = False,
                       caller: str = "") -> str:
    """默认全走 API；allow_local=True 且 API 失败时降级 Ollama 7B。

    费曼追问环节必须传 allow_local=False（7B 会盲目附和，降智不可接受），
    API 挂时 feynman.py 应明确提示"离线模式不支持追问"而不是降级。
    """
    text, expanded, usage, latency, source = _budget_retry_once(
        messages, temperature, max_tokens, allow_local=allow_local)
    if text:
        _fire_hook(caller, usage, latency, 1, expanded, source)
        return text
    # 重试（API 失败但降级成功也算成功；两者都失败才抛错）
    for attempt in range(2, MAX_RETRIES + 1):
        time.sleep(1.5 * (attempt - 1))
        text, expanded, usage, latency, source = _budget_retry_once(
            messages, temperature, max_tokens, allow_local=allow_local)
        if text:
            _fire_hook(caller, usage, latency, attempt, expanded, source)
            return text
    raise ModelError("LLM 调用失败（API 与本地降级均无有效输出）")


def _fire_hook(caller: str, usage: dict, latency_ms: int,
               attempts: int, expanded: bool, source: str) -> None:
    """触发日志 hook（若已注册）。usage 缺失时补 0。"""
    if not LOG_HOOK:
        return
    try:
        LOG_HOOK(
            caller=caller or "unknown",
            model=config.DEEPSEEK_MODEL,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            latency_ms=latency_ms,
            attempts=attempts,
            expanded=expanded,
            source=source,
            user_id=_CURRENT_USER.get(),
            session_id=_CURRENT_SESSION.get(),
        )
    except Exception as e:  # 日志失败绝不影响主流程
        print(f"[!] LLM 日志写入失败（忽略）: {type(e).__name__}: {e}")
