# -*- coding: utf-8 -*-
"""latency_bench.py — 性能基准：P50/P95 延迟（证据进 EVALUATION.md「性能基线」节）

两个口径（生成段与检索模式无关，accurate 只计检索+精排段，避免重复计生成）：
  fast     = 全链路（向量检索 + llama3.1:8b 生成）——用户-facing 快速模式的最坏路径
             （直接打 rag_core，绕过语义缓存；缓存命中 40ms 另见看板/亮点口径）
  accurate = 检索+精排段（向量 + BM25 + RRF + bge-reranker CPU）；总耗时 = 本口径
             + fast 口径中的生成段（两段独立可加）

用法：
  python latency_bench.py --mode fast --n 10
  python latency_bench.py --mode accurate --n 10
逐题追加 _latency_bench.csv 备查；stdout 打印本轮 P50/P90/P95/均值。
"""
import argparse
import csv
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_latency_bench.csv")

# 混合课程题面（覆盖笔记/教材主要主题；固定清单保证可复现）
QUERIES = [
    "什么是事务的 ACID 特性",
    "第三范式和 BCNF 的区别",
    "什么是死锁？四个必要条件",
    "进程和线程的区别",
    "TCP 三次握手的过程",
    "什么是动态规划",
    "Python 装饰器是什么",
    "HTTP 中 GET 和 POST 的区别",
    "事务的隔离级别有哪些",
    "二叉树有哪些遍历方式",
    "微信小程序的页面生命周期",
    "B 树和 B+ 树的区别",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "accurate"], required=True)
    ap.add_argument("--n", type=int, default=10, help="本轮题数（1-12）")
    args = ap.parse_args()
    n = max(1, min(12, args.n))
    queries = QUERIES[:n]

    from rag_core import build_rag_chain, retrieve_and_rerank

    chain = None
    if args.mode == "fast":
        chain = build_rag_chain()["chain"]  # 生成模型加载不计入单题耗时

    rows = []
    for i, q in enumerate(queries, 1):
        t0 = time.time()
        if args.mode == "fast":
            docs = retrieve_and_rerank(q, force_no_rerank=True)
            chain.invoke(q, docs=docs)  # 全链路：检索 + 生成
        else:
            retrieve_and_rerank(q, force_no_rerank=False)  # 仅检索 + 精排段
        ms = int((time.time() - t0) * 1000)
        rows.append(ms)
        print(f"[{i:>2}/{n}] {ms:>6} ms  {q}", flush=True)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [time.strftime("%Y-%m-%d %H:%M:%S"), args.mode, q, ms, len(rows)])

    rows.sort()
    mean = statistics.mean(rows)
    p = lambda q_: rows[max(0, min(len(rows) - 1, round(q_ / 100 * len(rows) + 0.5) - 1))]
    print(f"\n== {args.mode} (n={n}, {platform.platform()}) ==")
    print(f"P50={p(50)} ms  P90={p(90)} ms  P95={p(95)} ms  mean={mean:.0f} ms")


if __name__ == "__main__":
    main()
