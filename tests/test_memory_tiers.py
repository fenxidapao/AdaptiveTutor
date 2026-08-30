"""F 阶段记忆分层测试（PLAN 22.1）：timed 采样 / stats 百分位 / 分层语义。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import memory  # noqa: E402


def test_timed_records_and_stats():
    memory.reset()
    with memory.timed("L1"):
        time.sleep(0.002)
    with memory.timed("L3"):
        time.sleep(0.001)
    s = memory.stats()
    assert s["L1"]["count"] == 1 and s["L1"]["p50_ms"] >= 2
    assert s["L2"]["count"] == 0 and s["L2"]["p50_ms"] is None
    assert s["L3"]["p95_ms"] >= s["L3"]["p50_ms"] >= 1


def test_timed_records_on_exception():
    """异常路径也必须计入采样（finally 语义）。"""
    memory.reset()
    try:
        with memory.timed("L2"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert memory.stats()["L2"]["count"] == 1


def test_timed_unknown_tier_is_noop():
    memory.reset()
    with memory.timed("L99"):
        pass
    assert all(v["count"] == 0 for v in memory.stats().values())


def test_reset_clears():
    with memory.timed("L1"):
        pass
    memory.reset()
    assert memory.stats()["L1"]["count"] == 0


def test_percentile_bounds():
    """p50/p95 落在样本 min/max 之间（最近秩法边界不越界）。"""
    memory.reset()
    for _ in range(20):
        with memory.timed("L1"):
            time.sleep(0.0002)
    s = memory.stats()["L1"]
    assert s["count"] == 20
    assert 0 < s["p50_ms"] <= s["p95_ms"]
