"""
dashboard.py — CourseRAG 最小可观测看板（P1 / #15，2026-08-25）

读取 query_log.py 写入的 .query_log.jsonl（结构见 query_log.py），
纯 Python + Streamlit 统计，零新依赖：
  - KPI：累计 / 今日查询、平均延迟、拒答/未接地率
  - 延迟：快速 vs 精确 均值 / P95 + 时序折线
  - 模式分布：快速 / 精确 计数
  - 来源 Top：被引用最多的资料文件
  - 低质量清单：拒答 / 未接地 的查询
  - 反馈流：👍/👎 统计与明细

运行（PowerShell）：
  & $PY -m streamlit run dashboard.py
"""
import json
import os
from collections import Counter
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st

QUERY_LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", ".query_log.jsonl"))


def _parse_ts(rec):
    """兼容秒/毫秒两种 ts 格式；解析失败返回 None。"""
    try:
        return datetime.fromisoformat(rec.get("ts"))
    except Exception:
        return None


def _p95(vals):
    """纯 Python 95 分位（近似：排序后取 idx=0.95*n 处）。"""
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * 0.95))]


def load_records():
    """读取日志，逐行容错（坏行跳过，不阻塞看板）。"""
    if not QUERY_LOG_PATH.exists():
        return []
    rows = []
    with open(QUERY_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    st.set_page_config(page_title="CourseRAG 可观测看板", page_icon="📊", layout="wide")
    st.title("📊 CourseRAG 可观测看板")
    st.caption(f"数据源：{QUERY_LOG_PATH}（由 query_log.py 写入；本页只读不改日志）")

    rows = load_records()
    if not rows:
        st.info("暂无查询日志。先在 app.py 问几道题（或调 main.py /query），再回来刷新。")
        return

    # ---------- 时间范围过滤 ----------
    today = date.today()
    range_opt = st.sidebar.radio("时间范围", ["今天", "近 7 天", "全部"], index=1)
    if range_opt == "今天":
        start = datetime.combine(today, datetime.min.time())
    elif range_opt == "近 7 天":
        start = datetime.combine(today - timedelta(days=6), datetime.min.time())
    else:
        start = None

    kept = []
    for r in rows:
        t = _parse_ts(r)
        if start is not None and (t is None or t < start):
            continue
        kept.append((r, t))
    if not kept:
        st.info("所选时间范围内没有查询记录，换一个范围试试。")
        return

    recs = [r for r, _ in kept]
    n = len(recs)

    # ---------- KPI ----------
    today_n = sum(1 for _, t in kept if t and t.date() == today)
    lat = [r["latency_ms"] for r in recs
           if isinstance(r.get("latency_ms"), (int, float))]
    avg_lat = int(sum(lat) / len(lat)) if lat else 0
    reject_n = sum(1 for r in recs
                   if r.get("rejected") is True or r.get("grounded") is False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计查询", n)
    c2.metric("今日查询", today_n)
    c3.metric("平均延迟 (ms)", f"{avg_lat:,}" if lat else "—")
    c4.metric("拒答 / 未接地率", f"{reject_n / n:.0%}")

    st.divider()

    # ---------- 延迟：均值 / P95（分模式） ----------
    st.subheader("⏱️ 延迟")
    mode_stats = {}
    for mode in ("快速", "精确"):
        vals = [r["latency_ms"] for r in recs
                if r.get("mode") == mode and isinstance(r.get("latency_ms"), (int, float))]
        if vals:
            mode_stats[mode] = (int(sum(vals) / len(vals)), _p95(vals))
    if mode_stats:
        cols = st.columns(len(mode_stats))
        for col, (mode, (avg, p95)) in zip(cols, mode_stats.items()):
            col.metric(f"{mode} 平均", f"{avg:,} ms")
            col.metric(f"{mode} P95", f"{p95:,} ms")
    else:
        st.write("暂无延迟数据（需要 query_log 写入 latency_ms 后产生）。")

    # 时序折线：按日志顺序（x=记录序号），仅看有延迟的
    if lat:
        st.line_chart({"延迟 (ms)": [r["latency_ms"] for r in recs
                                     if isinstance(r.get("latency_ms"), (int, float))]})
        st.caption("按日志写入顺序展示（最新在右），x 轴为记录序号。")

    st.divider()

    # ---------- 模式分布 ----------
    st.subheader("🔀 模式分布")
    mode_cnt = Counter(r.get("mode") for r in recs)
    if mode_cnt:
        cols = st.columns(len(mode_cnt))
        for col, (m, c) in zip(cols, mode_cnt.most_common()):
            col.metric(m or "未知", c)
    else:
        st.write("无模式数据。")

    st.divider()

    # ---------- 来源 Top ----------
    st.subheader("📄 来源 Top")
    src_cnt = Counter()
    for r in recs:
        for s in (r.get("top_sources") or []):
            src = s.get("source")
            if src:
                src_cnt[src] += 1
    if src_cnt:
        st.table([{"资料": k, "被引用次数": v} for k, v in src_cnt.most_common(10)])
    else:
        st.write("暂无来源数据。")

    st.divider()

    # ---------- 低质量 / 拒答清单 ----------
    st.subheader("⚠️ 拒答 / 未接地查询")
    low = [r for r in recs if r.get("rejected") is True or r.get("grounded") is False]
    if low:
        st.table([{
            "时间": r.get("ts"),
            "问题": (r.get("question") or "")[:40],
            "模式": r.get("mode"),
            "延迟ms": r.get("latency_ms"),
            "拒答": "是" if r.get("rejected") else "",
            "反馈": r.get("feedback") or "",
        } for r in low[-20:]][::-1])  # 最新在前
    else:
        st.write("无（所选范围内没有拒答 / 未接地记录）。")

    st.divider()

    # ---------- 反馈流 ----------
    st.subheader("👍/👎 反馈")
    up = sum(1 for r in recs if r.get("feedback") == "👍")
    down = sum(1 for r in recs if r.get("feedback") == "👎")
    f1, f2 = st.columns(2)
    f1.metric("👍 有用", up)
    f2.metric("👎 不好", down)
    fb_rows = [r for r in recs if r.get("feedback")]
    if fb_rows:
        st.table([{
            "时间": r.get("ts"),
            "反馈": r.get("feedback"),
            "问题": (r.get("question") or "")[:40],
            "模式": r.get("mode"),
        } for r in fb_rows[-20:]][::-1])
    else:
        st.write("暂无反馈——在 app.py 的每条回答下方点 👍/👎 即可收集。")


if __name__ == "__main__":
    main()
