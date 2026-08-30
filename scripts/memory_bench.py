"""记忆分层延迟基准（F 阶段，PLAN 22.1 记忆分层行）：读路径延迟实测 + Little 定律推演。

回答一个问题：为什么本项目把记忆放进程内 SQLite 而不是网络存储（Redis 等）？
用本仓库真实读路径的实测延迟说话——

  L1 工作记忆（状态快照 build_snapshot，含 3 次 L3 读 + 策略规则）
  L2 会话记忆（render_transcript，短/长对话两档）
  L3 长期记忆（get_profile / get_kp / get_exercise_logs 单次读）

再按 Little 定律（在途请求数 L = 到达率 λ × 平均停留时间 W）推演：同样的业务
QPS 下，把记忆放到 5ms / 50ms 外的网络存储，在途请求会膨胀多少倍——
延迟不是体验问题，是并发容量问题。这就是"记忆的敌人首先是距离"的量化版。

纯本地、零网络、零 LLM 调用，可进 CI（快，~几秒）。

用法：
  /d/anacoda3/python.exe scripts/memory_bench.py [--out reports/memory_bench.md]
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
from agents import context  # noqa: E402
from agents import memory as mem  # noqa: E402

N_ROUNDS = 300  # 每项测量次数


def _seed(dbp: str) -> dict:
    """造一个有真实数据形态的用户：12 知识点 + 40 条答题日志 + 画像。"""
    db.init_db(dbp)
    uid = "u_bench"
    db.get_user(uid, db_path=dbp)
    for i in range(40):
        db.log_exercise(uid, f"python.ex.{i % 12}", f"python.kp.{i % 12}",
                        i % 3 != 0, "ans", "fb", db_path=dbp)
    db.save_profile(uid, {
        "weak_points": [{"kp_id": "python.kp.0", "reason": "r", "evidence": []}],
        "learning_style": "简答", "avg_correct": 0.55,
    }, dbp)
    kp = {"kp_id": "python.kp.1", "title": "测试知识点", "chapter": "ch1",
          "prerequisites": []}
    return {"uid": uid, "kp": kp}


def _bench(fn, n: int = N_ROUNDS) -> dict:
    """跑 n 次取延迟统计（毫秒）。"""
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return {"p50": round(statistics.median(xs), 3),
            "p95": round(xs[max(0, int(len(xs) * 0.95) - 1)], 3),
            "mean": round(statistics.fmean(xs), 3)}


def littles_law(rows: list) -> list:
    """同一业务 QPS 下不同记忆延迟的在途请求数（L = λ × W，W 取各方案 p95）。

    λ 取每层"该层读操作"的并发压力（读多路复用后远小于业务 QPS，
    取 λ=50 读/秒为示例量级——教育站单机全站 QPS 也就这个量级）。
    在途请求数直接决定：连接池大小/超时设置/线程或协程占用——
    50ms 记忆在 50 QPS 下常驻 2.5 个在途请求，5ms 只需 0.25，
    若叠加慢查询抖动到 500ms，就是 25 个——连接池打满、请求排队雪崩。
    rows: [(方案名, {"p50":..,"p95":..,"mean":..}), ...]，第一行为本地基线。
    """
    lam = 50  # 读/秒（示例量级）
    base_w = rows[0][1]["p95"] or 1.0
    out = []
    for label, r in rows:
        w = r["p95"]
        out.append({"方案": label, "延迟W_p95_ms": w,
                    "在途请求L(λ=50)": round(lam * w / 1000.0, 2),
                    "相对本地倍数": round(w / base_w, 1)})
    return out


def main():
    ap = argparse.ArgumentParser(description="记忆分层延迟基准（Little 定律）")
    ap.add_argument("--out", default="reports/memory_bench.md")
    args = ap.parse_args()

    mem.reset()
    # ignore_cleanup_errors：Windows 上 SQLite WAL 文件句柄释放晚于 rmtree
    # 是已知竞态（测量已完成，残留交给系统临时目录回收）
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        dbp = str(Path(td) / "bench.db")
        seeded = _seed(dbp)
        uid, kp = seeded["uid"], seeded["kp"]

        long_transcript = [{"role": "user" if i % 2 == 0 else "assistant",
                            "content": f"第{i}轮对话内容，切片讲解示例。"} for i in range(12)]

        rows = [
            ("L1 状态快照 build_snapshot（含策略规则+3次L3读）",
             _bench(lambda: context.build_snapshot(uid, kp, dbp, round_no=2))),
            ("L2 会话记忆 render_transcript（6 条短对话）",
             _bench(lambda: context.render_transcript(long_transcript[:6]))),
            ("L2 会话记忆 render_transcript（12 条长对话）",
             _bench(lambda: context.render_transcript(long_transcript))),
            ("L3 get_profile（长期记忆读）",
             _bench(lambda: db.get_profile(uid, dbp))),
            ("L3 get_kp（知识点状态读）",
             _bench(lambda: db.get_kp(uid, kp["kp_id"], dbp))),
            ("L3 get_exercise_logs（答题日志读，40 条）",
             _bench(lambda: db.get_exercise_logs(uid, db_path=dbp))),
        ]

        # memory.timed 采样验证（应与上面 L1 行同量级）
        for _ in range(N_ROUNDS):
            context.build_snapshot(uid, kp, dbp)
        timed_stats = mem.stats()

        # 对照组：模拟"记忆放远处"——同样的 L3 读加 5ms / 50ms 网络往返
        remote_rows = [
            ("对照：本地 L3 读（实测 p95）", rows[3][1]),
            ("对照：网络记忆 5ms（Redis 同机房典型）", {"p50": 5.0, "p95": 5.0, "mean": 5.0}),
            ("对照：网络记忆 50ms（跨可用区/公网典型）", {"p50": 50.0, "p95": 50.0, "mean": 50.0}),
        ]
        law = littles_law(remote_rows)

    lines = ["# 记忆分层延迟基准（Little 定律）", "",
             f"- 生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
             f"- 每项实测 {N_ROUNDS} 次，临时库 40 条答题日志 + 画像", "",
             "## 一、本项目读路径实测（毫秒）", "",
             "| 层 | 读路径 | p50 | p95 | mean |", "|---|---|---|---|---|"]
    for label, r in rows:
        tier = label.split(" ")[0]
        lines.append(f"| {tier} | {label} | {r['p50']} | {r['p95']} | {r['mean']} |")

    lines += ["", "## 二、memory.timed 采样（进程内观测自洽性）", ""]
    for tier, s in timed_stats.items():
        lines.append(f"- {tier}: count={s['count']}, p50={s['p50_ms']}ms, p95={s['p95_ms']}ms")

    lines += ["", "## 三、Little 定律推演：记忆放远处会怎样", "",
              "在途请求数 L = λ × W（λ=50 读/秒示例量级）。L 决定连接池/超时/线程占用——", "",
              "| 方案 | 延迟 W (ms) | 在途请求 L (λ=50) | 相对本地倍数 |", "|---|---|---|---|"]
    for r in law:
        lines.append(f"| {r['方案']} | {r['延迟W_p95_ms']} | {r['在途请求L(λ=50)']} | {r['相对本地倍数']} |")

    lines += [
        "", "## 结论", "",
        "1. 本项目全部记忆读路径（含快照的规则计算）p95 在亚毫秒~个位数毫秒量级，",
        "   单机 SQLite + WAL 足以支撑教育站真实并发（全站日限 300 次 LLM 调用的量级）；",
        "2. 若为'记忆分层'引入 50ms 网络存储，同 QPS 在途请求 ×量级放大，",
        "   连接池/超时全线吃紧——分层解决'注入什么'（信息熵减），",
        "   存储解决'读多快'（容量），两件事分开治理，本项目两层都要但都不需要 Redis；",
        "3. 阈值管理：若未来读路径 p95 超过 5ms（数据量增长/迁移上云），",
        "   再评估缓存层——延迟预算写在这里，超线先优化读路径而不是加层。",
    ]
    report = "\n".join(lines)
    print(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n[报告已写入 {out}]")


if __name__ == "__main__":
    main()
