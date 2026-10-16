"""model.py 单元测试：推理预算自动扩容逻辑（不依赖真实网络）。

deepseek-v4-flash 是推理模型，max_tokens 是"推理+回答"总预算。
本测试 mock _post_chat，验证：预算不足 → 自动翻倍扩容 → 最终成功。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402
from model import ReasoningBudgetError  # noqa: E402


def test_budget_expansion_retries(monkeypatch):
    """首次调用预算不足抛 ReasoningBudgetError，扩容后成功返回。"""
    calls = []

    def fake_post(messages, temperature, max_tokens, base_url, api_key, model):
        calls.append(max_tokens)
        if len(calls) == 1:  # 第一次：预算不足
            raise ReasoningBudgetError("预算不足", max_tokens=max_tokens)
        return "成功的回答", {"prompt_tokens": 10, "total_tokens": 20}, 50

    monkeypatch.setattr(model, "_post_chat", fake_post)
    text = model.chat([{"role": "user", "content": "hi"}], max_tokens=200)
    assert text == "成功的回答"
    assert calls == [200, 400]  # 翻倍扩容


def test_budget_expansion_caps_at_max(monkeypatch):
    """预算不断不足时，扩容到 MAX_BUDGET 封顶，不无限翻倍。"""
    calls = []

    def fake_post(messages, temperature, max_tokens, base_url, api_key, model):
        calls.append(max_tokens)
        raise ReasoningBudgetError("预算不足", max_tokens=max_tokens)

    monkeypatch.setattr(model, "_post_chat", fake_post)
    with pytest.raises(model.ModelError):
        model.chat([{"role": "user", "content": "hi"}], max_tokens=200)
    assert max(calls) == model.MAX_BUDGET  # 封顶 8000
    # 每次外层重试：200→400→800→1600→3200→6400→8000 共 7 次扩容调用；外层 3 次重试
    assert len(calls) == 3 * 7


def test_network_error_no_expansion(monkeypatch):
    """非预算错误（网络/HTTP）不触发扩容，正常重试后抛错。"""
    calls = []

    def fake_post(messages, temperature, max_tokens, base_url, api_key, model):
        calls.append(max_tokens)
        import urllib.error
        raise urllib.error.URLError("网络挂了")

    monkeypatch.setattr(model, "_post_chat", fake_post)
    with pytest.raises(model.ModelError):
        model.chat([{"role": "user", "content": "hi"}], max_tokens=200)
    assert all(c == 200 for c in calls)  # 不扩容


def test_reasoning_error_has_max_tokens_attr():
    """ReasoningBudgetError 携带 max_tokens 信息。"""
    e = model.ReasoningBudgetError("预算不足", max_tokens=800)
    assert e.max_tokens == 800
    assert isinstance(e, model.ModelError)
