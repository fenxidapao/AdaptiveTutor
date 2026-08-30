"""记忆分层（F 阶段，PLAN 22）：三层记忆 + 每层访问延迟可观测。

三层职责（信息熵减：每层只回答一个问题，模型不在历史里大海捞针）：
  L1 工作记忆 = 状态快照。确定性函数从状态库拼装"当前知识点/掌握度/连错数/
     建议策略/薄弱点"的干净视图（context.build_snapshot），每轮注入 prompt——
     模型决策只依赖快照，不依赖考古。状态变更不经过模型：答题 → 规则判题 →
     规则画像（db.update_profile_incremental），模型只产生文本。
  L2 会话记忆 = transcript + 压缩摘要（context.render_transcript/compress_old），
     超长自动压缩（②Context），最近原文保留供追问指向。
  L3 长期记忆 = 状态库（profile/knowledge_points/reflow_logs/exercise_logs），
     跨会话持久，全部带 user_id。

为什么不用 Redis List/Hash + 分布式锁（行业常见方案）：那是多实例部署下
"记忆放远处"的代价管理方案。本项目单机 SQLite，读路径实测亚毫秒
（见 scripts/memory_bench.py）；而把记忆挪到 50ms 外的网络存储，同样 QPS 下
在途请求数会按 Little 定律放大一个数量级——记忆的敌人首先是距离，其次才是层次。
并发写安全由 db._conn 的 WAL+busy_timeout（引擎排队）与 /api/grade 的
request_id 幂等（应用层去重）承担，不需要应用级分布式锁。

延迟预算（Little 定律：在途请求数 = 到达率 × 平均停留时间）：记忆访问延迟
不是体验问题而是容量问题——5ms→50ms，同 QPS 在途请求 ×10，连接池/超时/
线程模型全线吃紧。timed()/stats() 让每层读取延迟可观测：p95 超预算先于
用户报障被发现。
"""

import time
from contextlib import contextmanager

_TIERS = ("L1", "L2", "L3")
_samples: dict[str, list[float]] = {t: [] for t in _TIERS}


@contextmanager
def timed(tier: str):
    """记录一次某层记忆访问耗时（毫秒）。tier ∈ L1/L2/L3，未知层忽略。

    用法：with memory.timed("L3"): prof = db.get_profile(uid)
    进程内聚合（不落库）——观测对象是"读路径延迟"，量级判断够用；
    持久化监控属于部署期可选项，不在教学链路上加写放大。
    """
    if tier not in _TIERS:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _samples[tier].append((time.perf_counter() - t0) * 1000.0)


def _pct(sorted_xs: list[float], p: int) -> float:
    """百分位数（最近秩法）：p=50 取中位，p=95 取 95 分位。"""
    idx = max(0, min(len(sorted_xs) - 1,
                     int(round(p / 100 * len(sorted_xs) + 0.5)) - 1))
    return sorted_xs[idx]


def stats() -> dict:
    """各层访问延迟统计：count / p50_ms / p95_ms（毫秒）。空层为 None。"""
    out: dict[str, dict] = {}
    for tier in _TIERS:
        xs = sorted(_samples.get(tier, []))
        if not xs:
            out[tier] = {"count": 0, "p50_ms": None, "p95_ms": None}
            continue
        out[tier] = {
            "count": len(xs),
            "p50_ms": round(_pct(xs, 50), 3),
            "p95_ms": round(_pct(xs, 95), 3),
        }
    return out


def reset() -> None:
    """清空采样（测试/基准隔离用）。"""
    for tier in _TIERS:
        _samples[tier] = []
