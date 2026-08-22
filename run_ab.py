"""
run_ab.py — 混合检索 A/B 断点续跑 runner（2026-08-21 新增）

两步顺序执行，每步循环调用 eval_local.py 直到该阶段全部样本完成：
  第 1 步  纯向量基线   HYBRID_ENABLED=false  -> eval_results_vector.csv
  第 2 步  混合检索     HYBRID_ENABLED=true   -> eval_results_hybrid.csv
  第 3 步  compare_hybrid.py 出对比报告（eval_compare_hybrid.md）

断点续跑机制（应对中断/断电/后台回收）：
  - eval_local.py 本身按 question 跳过已完成、每题增量落盘（已有能力）；
  - 本 runner 每次 attempt 前检查 CSV 已完成题数，未满则继续跑（自动续上）；
  - 中断后重新执行 `python run_ab.py` 即可从断点继续，无需从头。
  - 每阶段日志追加到 eval_ab_<stage>.log，便于事后排查。

隔离要求（关键）：
  - 强制 RERANKER_ENABLED=false 覆盖 .env 的 true —— 两轮都关 reranker，
    才能隔离出「BM25 词面召回」的纯贡献；否则 reranker 会盖掉 BM25 效应。
  - 两轮用不同 EVAL_CSV_NAME，避免 eval_local 的断点续跑把第二轮当已完成跳过。

用法：
  python run_ab.py
"""
import os
import csv
import sys
import subprocess

PY = r"C:\Users\user\.workbuddy\binaries\python\envs\evalverify\Scripts\python.exe"
ROOT = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = 100  # eval_data.py 当前样本数；改样本后同步改这里

STAGES = [
    {"name": "vector", "csv": "eval_results_vector.csv",
     "env": {"HYBRID_ENABLED": "false"}},
    {"name": "hybrid", "csv": "eval_results_hybrid.csv",
     "env": {"HYBRID_ENABLED": "true"}},
]

MAX_ATTEMPTS = 200  # 每阶段最多尝试轮数（防死循环；正常一轮就够）


def done_count(csv_path):
    """CSV 中去重 question 数（未完成/中断时按实际落盘统计）。"""
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            qs = {r["question"] for r in csv.DictReader(f) if r.get("question")}
        return len(qs)
    except Exception:
        return 0


def run_stage(stage):
    csv_path = os.path.join(ROOT, stage["csv"])
    log_path = os.path.join(ROOT, f"eval_ab_{stage['name']}.log")
    env = os.environ.copy()
    env.update(stage["env"])
    env["RERANKER_ENABLED"] = "false"      # 关键：隔离 BM25 贡献
    env["EVAL_CSV_NAME"] = stage["csv"]
    env["SKIP_REBUILD"] = "1"              # A/B 两阶段共用同一索引（避免每次启动写库崩 SQLite）

    for attempt in range(1, MAX_ATTEMPTS + 1):
        done = done_count(csv_path)
        print(f"[{stage['name']}] attempt {attempt}: 已完成 {done}/{N_SAMPLES}", flush=True)
        if done >= N_SAMPLES:
            print(f"[{stage['name']}] 全部完成 ✓", flush=True)
            return True
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {stage['name']} attempt {attempt} start "
                     f"(ts={__import__('datetime').datetime.now().isoformat(timespec='seconds')}) =====\n")
            proc = subprocess.run(
                [PY, "eval_local.py"], cwd=ROOT, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            # 中断/异常退出：eval_local 已把已完成题落盘，下一轮 attempt 自动续跑
            print(f"[{stage['name']}] eval_local 退出码 {proc.returncode}，断点续跑中…", flush=True)
    print(f"[{stage['name']}] 超过 {MAX_ATTEMPTS} 次尝试仍未完成，请检查 {log_path}", flush=True)
    return False


if __name__ == "__main__":
    print("=== 混合检索 A/B runner（断点续跑）===", flush=True)
    print(f"Python: {PY}", flush=True)
    print(f"样本数: {N_SAMPLES}（若 eval_data.py 有改动请同步 N_SAMPLES）", flush=True)
    for stage in STAGES:
        ok = run_stage(stage)
        if not ok:
            sys.exit(2)
    print("\n两阶段完成，运行 compare_hybrid.py 生成报告…", flush=True)
    subprocess.run(
        [PY, "compare_hybrid.py",
         os.path.join(ROOT, "eval_results_vector.csv"),
         os.path.join(ROOT, "eval_results_hybrid.csv")],
        cwd=ROOT,
    )
    print("=== 全部完成，报告见 eval_compare_hybrid.md ===", flush=True)
