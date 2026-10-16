"""
feedback_pool.py — #17 完整反馈闭环：把用户 👎 回流成评估集（B/C 组）

闭环链路（从 app.py 的 👍/👎 开始）：
  用户 👎 → .query_log.jsonl（query_log.set_feedback 已写入）
  → `list`：从 jsonl 提取 👎 候选 → eval_feedback_pool.json（按 question 去重，status=pending）
  → `promote`：人工定标后移入 eval_data_feedback.json（B 或 C 组）
  → `eval_heldout.py --bc-only`：合并回流用例重新门控（B 组拒答率 ≥90%）

设计原则：
  - 不自动改 HELDOUT_A/B/C 源码常量——2026-08-24 冻结基线不被污染（#18 评估纪律）；
  - 回流用例走独立 JSON，eval_data_heldout 运行时合并，便于审计 / 撤销；
  - 👎 只是"候选信号"，是否进评估集由人定标（--expect）——避免把噪声当金标准。

用法：
  python feedback_pool.py list
  python feedback_pool.py promote <idx> --group B            # 语料外应拒答
  python feedback_pool.py promote <idx> --group C --expect reject|answer
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

QUERY_LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", ".query_log.jsonl"))
POOL_PATH = Path(os.getenv("FEEDBACK_POOL_PATH", "eval_feedback_pool.json"))
FEEDBACK_JSON = Path(os.getenv("FEEDBACK_JSON", "eval_data_feedback.json"))


def load_records():
    """读取查询日志，逐行容错。"""
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


def load_pool():
    if POOL_PATH.exists():
        try:
            return json.loads(POOL_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_pool(pool):
    POOL_PATH.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_feedback_json():
    if FEEDBACK_JSON.exists():
        try:
            return json.loads(FEEDBACK_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_feedback_json(cases):
    FEEDBACK_JSON.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_list():
    """从查询日志同步 👎 候选到候选池（去重），并打印待定清单。"""
    rows = load_records()
    down = [r for r in rows if r.get("feedback") == "👎"]
    pool = load_pool()
    # 已回流过的问题不再进池：既查评估集 JSON，也查池内已 promote 的
    promoted_qs = {c.get("question") for c in load_feedback_json()}
    seen = {p.get("question") for p in pool if p.get("status") == "promoted"}
    existing = {p.get("question") for p in pool}

    added = 0
    for r in down:
        q = r.get("question", "")
        if not q or q in seen or q in promoted_qs or q in existing:
            continue
        pool.append({
            "question": q,
            "ts": r.get("ts"),
            "mode": r.get("mode"),
            "grounded": r.get("grounded"),
            "rejected": r.get("rejected"),
            "latency_ms": r.get("latency_ms"),
            "answer_len": r.get("answer_len"),
            "top_sources": r.get("top_sources") or [],
            "status": "pending",
        })
        existing.add(q)  # 同一批次内的同问句也去重（existing 是循环前快照，需手动更新）
        added += 1
    save_pool(pool)

    pending = [p for p in pool if p.get("status") == "pending"]
    print(f"👎 反馈记录：{len(down)} 条；本次新增候选：{added}")
    if not pending:
        print("（候选池无待定项）")
        return
    print(f"待定 {len(pending)} 条，人工定标后用 promote 回流：")
    for i, p in enumerate(pending):
        print(f"  [{i}] {p['question'][:50]}  mode={p.get('mode')}  "
              f"grounded={p.get('grounded')}  rejected={p.get('rejected')}")


def cmd_promote(idx, group, expect):
    """把池内第 idx 条候选定标后移入评估集 JSON（B 或 C 组）。"""
    pool = load_pool()
    pending = [p for p in pool if p.get("status") == "pending"]
    if not (0 <= idx < len(pending)):
        print(f"无效索引 {idx}（可用范围 0~{len(pending) - 1}）")
        sys.exit(2)
    p = pending[idx]

    group = (group or "").upper()
    if group not in ("B", "C"):
        print("group 必须是 B 或 C")
        sys.exit(2)
    if group == "B":
        expect = True  # B 组 = 语料外应拒答
    elif expect is None:
        print("C 组需显式 --expect reject|answer")
        sys.exit(2)
    else:
        expect = expect.lower() == "reject"

    cases = load_feedback_json()
    cases.append({
        "question": p["question"],
        "expect_reject": expect,
        "group": group,
        "note": "用户 👎 反馈回流（#17 闭环）",
        "source_ts": p.get("ts"),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_feedback_json(cases)

    p["status"] = "promoted"
    p["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    save_pool(pool)

    print(f"已回流 → {group} 组（expect_reject={expect}）：{p['question'][:50]}")
    print(f"评估集现有回流用例：{len(cases)} 条（{FEEDBACK_JSON}）")
    print("重跑门控验证：python eval_heldout.py --bc-only")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "list":
        cmd_list()
    elif args[0] == "promote":
        if len(args) < 2:
            print("用法：python feedback_pool.py promote <idx> --group B|C [--expect reject|answer]")
            sys.exit(2)
        idx = int(args[1])
        group = expect = None
        i = 2
        while i < len(args):
            if args[i] == "--group" and i + 1 < len(args):
                group = args[i + 1]
                i += 2
            elif args[i] == "--expect" and i + 1 < len(args):
                expect = args[i + 1]
                i += 2
            else:
                i += 1
        cmd_promote(idx, group, expect)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
