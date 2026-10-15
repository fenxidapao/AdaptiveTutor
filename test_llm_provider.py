"""test_llm_provider.py — C3 生成去本地化的确定性测试（不联网、不依赖 Ollama）。

覆盖：
  1. provider 分发：默认 ollama 行为不变；openai 走云端 HTTP。
  2. key 解析优先级与「找不到 key」的失败文案（不泄漏 key）。
  3. 推理模型预算自适应：content 为空且 reasoning 占满预算时翻倍重试。
  4. 云端流式 SSE：只取 delta.content，跳过 reasoning_content。
  5. 云端异常（HTTP 4xx/5xx、网络不可达）转成 LLMError，不抛 urllib 原始异常。
  6. 生成链路末端：rag_core.ollama_chat / ollama_chat_stream 仍可被 monkeypatch
     （test_api.py 四处钉的就是这两个名字）。
"""
import io
import json

import pytest

import config
import llm_provider
import rag_core


# ====================== 1. provider 分发 ======================
def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    called = {}

    def fake_ollama(messages, temperature):
        called["hit"] = True
        called["temperature"] = temperature
        return "本地回答"

    monkeypatch.setattr(llm_provider, "_ollama_chat", fake_ollama)
    out = llm_provider.chat([{"role": "user", "content": "hi"}], temperature=0.1)
    assert out == "本地回答"
    assert called["hit"] is True and called["temperature"] == 0.1


def test_openai_provider_uses_cloud(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    seen = {}

    def fake_cloud(messages, temperature, max_tokens):
        seen["model"] = config.LLM_MODEL
        return "云端回答"

    monkeypatch.setattr(llm_provider, "_cloud_chat", fake_cloud)
    out = llm_provider.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert out == "云端回答"
    assert seen["model"] == config.LLM_MODEL


def test_unknown_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "some-typo")
    monkeypatch.setattr(llm_provider, "_ollama_chat", lambda m, t: "fallback")
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "fallback"


def test_rag_core_delegates_to_provider(monkeypatch):
    """rag_core.ollama_chat 必须委托给 provider，而不是自己直连 Ollama。"""
    monkeypatch.setattr(llm_provider, "chat", lambda messages, temperature=None: "via-provider")
    assert rag_core.ollama_chat([{"role": "user", "content": "x"}]) == "via-provider"

    monkeypatch.setattr(
        llm_provider, "chat_stream",
        lambda messages, temperature=None: iter(["a", "b"]),
    )
    assert list(rag_core.ollama_chat_stream([{"role": "user", "content": "x"}])) == ["a", "b"]


def test_rag_core_functions_remain_monkeypatchable(monkeypatch):
    """test_api.py 四处钉 rag_core.ollama_chat 名字，改名会连带破坏测试。"""
    monkeypatch.setattr(rag_core, "ollama_chat", lambda messages, temperature=None: "patched")
    assert rag_core.ollama_chat([{"role": "user", "content": "x"}]) == "patched"


# ====================== 2. key 解析 ======================
def test_resolve_key_prefers_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    assert llm_provider.resolve_api_key() == "env-key"


def test_resolve_key_falls_back_to_credentials_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("KUAIPAO_API_KEY", raising=False)
    monkeypatch.delenv("VISUAL_RERANK_API_KEY", raising=False)
    cred = tmp_path / ".credentials.yaml"
    cred.write_text("other: x\nKUAIPAO_API_KEY: file-key\n", encoding="utf-8")
    monkeypatch.setattr(llm_provider, "_CRED_FILE", cred)
    assert llm_provider.resolve_api_key() == "file-key"


def test_missing_key_raises_without_leaking(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("KUAIPAO_API_KEY", raising=False)
    monkeypatch.delenv("VISUAL_RERANK_API_KEY", raising=False)
    monkeypatch.setattr(llm_provider, "_CRED_FILE", tmp_path / "absent.yaml")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    msg = str(exc.value)
    assert "API key" in msg
    # 失败文案里不能出现任何看起来像 key 的字符串
    assert "sk-" not in msg


# ====================== 3. 推理预算自适应 ======================
def _cloud_body(content="", reasoning="", finish="length"):
    return {
        "choices": [{
            "message": {"content": content, "reasoning_content": reasoning},
            "finish_reason": finish,
        }]
    }


def test_cloud_chat_auto_expands_on_reasoning_only(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 256)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS_CAP", 2048)
    budgets = []

    def fake_post(messages, temperature, max_tokens, **_kwargs):
        budgets.append(max_tokens)
        if max_tokens < 1024:
            return _cloud_body(content="", reasoning="想" * max_tokens, finish="length")
        return _cloud_body(content="最终答案", reasoning="想", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    out = llm_provider.chat([{"role": "user", "content": "x"}])
    assert out == "最终答案"
    assert budgets == [256, 512, 1024]   # 每次翻倍


def test_cloud_chat_raises_when_cap_exhausted(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 256)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS_CAP", 512)

    def fake_post(messages, temperature, max_tokens, **_kwargs):
        return _cloud_body(content="", reasoning="想" * max_tokens, finish="length")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "LLM_MAX_TOKENS" in str(exc.value)


def test_cloud_chat_disables_thinking_by_default(monkeypatch):
    """默认必须发送关闭思维链的参数：实测 reasoning 会烧光预算导致空 content。"""
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", True)
    seen = []

    def fake_post(messages, temperature, max_tokens, thinking_off=False, thinking_style=0):
        seen.append((thinking_off, thinking_style))
        return _cloud_body(content="答案", reasoning="", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "答案"
    assert seen and seen[0][0] is True      # thinking_off=True


def test_cloud_chat_can_keep_thinking_when_disabled(monkeypatch):
    """LLM_DISABLE_THINKING=false 时不注入该参数（保留原行为，便于对比）。"""
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", False)
    seen = []

    def fake_post(messages, temperature, max_tokens, thinking_off=False, thinking_style=0):
        seen.append(thinking_off)
        return _cloud_body(content="答案", reasoning="", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    llm_provider.chat([{"role": "user", "content": "x"}])
    assert seen == [False]


def test_thinking_off_payload_variants_exist():
    """至少提供一个关闭思考的字段，且每个都是 dict（供 payload.update）。"""
    assert len(llm_provider._THINKING_OFF_ATTEMPTS) >= 1
    for item in llm_provider._THINKING_OFF_ATTEMPTS:
        assert isinstance(item, dict) and item

def test_cloud_chat_returns_content_when_present(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        llm_provider, "_cloud_post",
        lambda m, t, mt, **kw: _cloud_body(content="答案", reasoning="想", finish="stop"),
    )
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "答案"


# ====================== 4. 云端流式 SSE ======================
class _FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self._lines:
            yield (line + "\n").encode("utf-8")


def _sse_line(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish}]}, ensure_ascii=False
    )


def test_cloud_stream_yields_only_content(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    lines = [
        _sse_line(reasoning="思考中"),
        _sse_line(content="你"),
        _sse_line(content="好"),
        _sse_line(reasoning="继续思考"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        llm_provider.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeStreamResp(lines),
    )
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")
    out = list(llm_provider.chat_stream([{"role": "user", "content": "x"}]))
    assert out == ["你", "好"]          # reasoning_content 不泄漏给用户


def test_cloud_stream_skips_malformed_frames(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    lines = [
        "data: {not json}",
        _sse_line(content="ok"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        llm_provider.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeStreamResp(lines),
    )
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")
    assert list(llm_provider.chat_stream([{"role": "user", "content": "x"}])) == ["ok"]


# ====================== 5. 云端异常 ======================
def test_cloud_http_error_becomes_llm_error(monkeypatch):
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error":"no access"}')
        )

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "403" in str(exc.value)


def test_cloud_url_error_becomes_llm_error(monkeypatch):
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")

    def boom(req, timeout=None):
        raise urllib.error.URLError("dns fail")

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError):
        llm_provider.chat([{"role": "user", "content": "x"}])


def test_no_key_in_error_when_http_body_echoes_key(monkeypatch):
    """即使上游把 key 回显在错误体里，也不能原样抛出（防日志泄漏）。"""
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "sk-SECRET-DO-NOT-LEAK")

    def boom(req, timeout=None):
        body = b'{"error":"bad key sk-SECRET-DO-NOT-LEAK"}'
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(body))

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "sk-SECRET-DO-NOT-LEAK" not in str(exc.value)


# ====================== 6. provider 名称解析 ======================
@pytest.mark.parametrize("raw,expected", [
    ("openai", "openai"), ("OPENAI", "openai"), (" openai ", "openai"),
    ("ollama", "ollama"), ("", "ollama"), ("typo", "ollama"),
])
def test_provider_name_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr(config, "LLM_PROVIDER", raw)
    assert llm_provider.provider_name() == expected