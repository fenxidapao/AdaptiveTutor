"""②Context 长对话压缩测试（9 步框架，PLAN 21 Pi 双层循环预留）：

- should_compress / render_transcript：阈值判断与摘要+原文渲染；
- compress_old：LLM 摘要 + 失败规则兜底；
- generate_followup：超长 transcript 自动压缩（prompt 含摘要、不含最旧原文）；
- feynman_round：增量压缩只调一次 LLM（复用 summary 不重复烧钱）。
纯规则层可单测，LLM 打桩。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import db  # noqa: E402
import model  # noqa: E402
from agents import context, feynman  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path):
    p = str(tmp_path / "state.db")
    db.init_db(p)
    return p


def _transcript(n: int) -> list[dict]:
    """构造 n 条交替对话（n 为偶数：user/coach 成对）。"""
    t = []
    for i in range(n // 2):
        t.append({"role": "user", "content": f"学生第{i}轮讲解内容{i}"})
        t.append({"role": "assistant", "content": f"教练追问{i}"})
    return t


# ==================== 规则层：阈值与渲染 ====================


def test_should_compress_threshold():
    """超过 CONTEXT_COMPRESS_AFTER 条才压缩（当前 3 轮=6 条不触发）。"""
    assert context.should_compress(_transcript(6)) is False
    assert context.should_compress(_transcript(config.CONTEXT_COMPRESS_AFTER)) is False
    assert context.should_compress(_transcript(config.CONTEXT_COMPRESS_AFTER + 2)) is True


def test_render_transcript_full():
    """正常路径：全量渲染，不压缩。"""
    t = _transcript(4)
    s = context.render_transcript(t)
    assert "学生第0轮讲解内容0" in s
    assert "教练追问1" in s


def test_render_transcript_with_summary():
    """压缩路径：摘要 + 最近 CONTEXT_RAW_TAIL 条原文，不含旧消息。"""
    t = _transcript(12)
    s = context.render_transcript(t, summary="学生讲对了基础，切片边界理解含糊")
    assert "【早期对话摘要】" in s
    assert "切片边界理解含糊" in s
    # 最近 4 条原文保留（第 9~11 条）
    assert "学生第4轮讲解内容4" in s
    # 最早的旧消息不进 prompt
    assert "学生第0轮讲解内容0" not in s
    assert "教练追问0" not in s


# ==================== LLM 摘要与兜底 ====================


def test_compress_old_returns_summary(monkeypatch, tmp_db):
    """compress_old：旧消息交给 LLM 压缩成摘要。"""
    def fake_chat(messages, **kw):
        assert "context_compress" in kw.get("caller", "")
        return "学生讲了变量与赋值，教练指出作用域理解有盲点"
    monkeypatch.setattr(model, "chat", fake_chat)
    t = _transcript(12)
    s = context.compress_old(t, db_path=tmp_db)
    assert "作用域" in s


def test_compress_old_fallback_on_error(monkeypatch, tmp_db):
    """LLM 故障：规则兜底文案（不阻塞、不抛错）。"""
    def boom(*a, **k):
        raise model.ModelError("故障")
    monkeypatch.setattr(model, "chat", boom)
    s = context.compress_old(_transcript(12), db_path=tmp_db)
    assert "已省略" in s


# ==================== 集成：generate_followup / feynman_round ====================


def test_followup_compresses_long_transcript(monkeypatch, tmp_db):
    """超长 transcript：prompt 含摘要与最近原文，不含最旧消息。"""
    captured = {}

    def fake_chat(messages, **kw):
        if kw.get("caller") == "context_compress":
            return "学生基础部分讲得对，切片边界易错"
        captured["user"] = messages[-1]["content"]
        return "追问：切片 end 含不含？"

    monkeypatch.setattr(model, "chat", fake_chat)
    kp = {"kp_id": "python.list.slice", "title": "列表切片"}
    feynman.generate_followup(kp, _transcript(12), db_path=tmp_db)
    assert "【早期对话摘要】" in captured["user"]
    assert "切片边界易错" in captured["user"]
    assert "学生第0轮讲解内容0" not in captured["user"]  # 最旧消息已压缩


def test_followup_short_transcript_no_compression(monkeypatch, tmp_db):
    """正常 3 轮（6 条）：不触发压缩。"""
    captured = {}

    def fake_chat(messages, **kw):
        if kw.get("caller") == "context_compress":
            raise AssertionError("不应触发压缩")
        captured["user"] = messages[-1]["content"]
        return "追问一下"

    monkeypatch.setattr(model, "chat", fake_chat)
    kp = {"kp_id": "python.list.slice", "title": "列表切片"}
    feynman.generate_followup(kp, _transcript(6), db_path=tmp_db)
    assert "【早期对话摘要】" not in captured["user"]
    assert "学生第0轮讲解内容0" in captured["user"]  # 全量


def test_feynman_round_incremental_compress_once(monkeypatch, tmp_db, capsys):
    """feynman_round 长轮数：增量压缩只调一次 LLM（复用 summary）。"""
    import json

    user = "u_ctx1"
    db.get_user(user, db_path=tmp_db)
    compress_calls = {"n": 0}

    def fake_chat(messages, **kw):
        caller = kw.get("caller", "")
        if caller == "context_compress":
            compress_calls["n"] += 1
            return "学生基础概念正确，切片步长理解有盲点"
        if caller == "feynman_gaps":
            return json.dumps({"gaps": ["切片步长"]}, ensure_ascii=False)
        return "教练追问"

    monkeypatch.setattr(model, "chat", fake_chat)
    # 10 轮 → transcript 累计 20 条 > 8，压缩应触发且只触发一次
    res = feynman.feynman_round(
        user, "python.list.slice", max_rounds=10,
        ask_user=lambda p: "变量是名字，切片是取子集",
        db_path=tmp_db,
    )
    assert compress_calls["n"] == 1  # 增量压缩：只压一次
    assert len(res["transcript"]) >= 10
