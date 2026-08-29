"""C 阶段实验分析（PLAN 18.3 / EXPERIMENT.md §6）。

读实验库 → 组间对比（feynman vs lecture 前后测提升）→ markdown 报告。
用法：/d/anacoda3/python.exe scripts/analyze_experiment.py [--db data/state.db] [--out reports/experiment.md]
默认读 data/state.db（容器卷挂载的实验库；2026-08-29 踩坑：曾默认读到根目录
pytest 残渣库 state.db，443 个假用户）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402

# 测试账号剔除（C2 运行约定，docs/EXPERIMENT_LOG.md）：非参与者数据不入组间统计/流失率
EXCLUDED_USER_PREFIXES = ("test_",)
EXCLUDED_USERS = {"smoke_test", "u0", "u_eval", "u_eval_mem"}


def _latest(rows, kind: str):
    """该用户 kind 的最新一条测评（按 created_at 排序取最后）。"""
    got = [r for r in rows if r["kind"] == kind]
    return got[-1] if got else None


def _first(rows, kind: str):
    """该用户 kind 的第一条测评。

    E1 学习回流（PLAN 20.2）会给同一用户追加多条 posttest 记录——
    组间对照必须取**第一条**（教学完成后的首次测量），回流重测单独统计，
    否则回流刷分会污染"费曼 vs 讲解"的对照数据。
    """
    got = [r for r in rows if r["kind"] == kind]
    return got[0] if got else None


def analyze(db_path: str) -> dict:
    all_users = {u["user_id"]: u for u in db.list_users(db_path)}
    users = {uid: u for uid, u in all_users.items()
             if not uid.startswith(EXCLUDED_USER_PREFIXES) and uid not in EXCLUDED_USERS}
    excluded = sorted(set(all_users) - set(users))
    rows_by_user = {}
    for uid in users:
        rows_by_user[uid] = db.list_assessments(uid, db_path=db_path)

    records = []  # 每个有前后测数据的用户一条
    dropouts = []  # 注册但无 posttest
    reflow = {"completed": 0, "failed": 0, "given_up": 0, "open": 0,
              "retest_gains": []}  # E1 回流统计（次要指标，PLAN 20.5）
    for uid, u in users.items():
        rows = rows_by_user[uid]
        pre = _latest(rows, "pretest")
        post = _first(rows, "posttest")  # E1 回流追加 posttest → 取第一条防污染
        if post is None:
            dropouts.append(uid)
            continue
        if pre is None:
            pre = {"score": None, "total": 0, "elapsed_seconds": None}
        gain = None
        if pre["score"] is not None:
            gain = round((post["score"] - pre["score"]) * 100, 1)
        records.append({
            "user_id": uid,
            "group": u.get("group_name") or "unknown",
            "pre": pre["score"],
            "post": post["score"],
            "gain_pp": gain,
            "elapsed": post.get("elapsed_seconds"),
            "total": post.get("total") or 0,
        })
        for r in db.list_reflows(uid, db_path):
            st = r["status"]
            if st in reflow:
                reflow[st] += 1
            if r.get("retest_score") is not None and r.get("trigger_score") is not None:
                reflow["retest_gains"].append(r["retest_score"] - r["trigger_score"])

    def flags(r):
        fl = []
        if r["pre"] is not None and r["pre"] >= 0.9:
            fl.append("天花板")
        if r["elapsed"] is not None and (r["elapsed"] < 15 or r["elapsed"] < r["total"] * 3):
            fl.append("疑似快速作答")
        return fl

    return {"records": records, "dropouts": dropouts, "flags": flags,
            "reflow": reflow, "excluded": excluded}


def _group_stats(records, include):
    """分组汇总。include(record)->bool 过滤。"""
    stats = {}
    for g in ("feynman", "lecture"):
        rows = [r for r in records if r["group"] == g and include(r)
                and r["pre"] is not None and r["gain_pp"] is not None]
        if not rows:
            stats[g] = None
            continue
        pre = sum(r["pre"] for r in rows) / len(rows)
        post = sum(r["post"] for r in rows) / len(rows)
        gain = sum(r["gain_pp"] for r in rows) / len(rows)
        stats[g] = {"n": len(rows), "pre": pre * 100, "post": post * 100,
                    "gain_pp": round(gain, 1)}
    return stats


def render(analysis: dict) -> str:
    records, dropouts, flags = analysis["records"], analysis["dropouts"], analysis["flags"]
    flag_map = {r["user_id"]: flags(r) for r in records}

    def all_incl(r):
        return True

    def strict_incl(r):
        return not flag_map[r["user_id"]]  # 剔除天花板 + 疑似快速作答

    excluded = analysis.get("excluded", [])
    lines = ["# C 阶段实验分析报告", "",
             f"- 生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
             f"- 剔除测试账号：{len(excluded)} 个（{', '.join(excluded) or '无'}；非参与者，约定见 EXPERIMENT_LOG）",
             f"- 注册用户数：{len(records) + len(dropouts)}，有前后测：{len(records)}，流失（无后测）：{len(dropouts)}",
             f"- 流失率：{len(dropouts) / (len(records) + len(dropouts)) * 100:.0f}%", ""]

    for label, incl in (("全部样本", all_incl), ("剔除天花板+疑似作弊", strict_incl)):
        s = _group_stats(records, incl)
        lines += [f"## {label}", "",
                  "| 组 | n | 前测均值 | 后测均值 | 提升均值(pp) |",
                  "|---|---|---|---|---|"]
        for g in ("feynman", "lecture"):
            v = s[g]
            if v is None:
                lines.append(f"| {g} | 0 | - | - | - |")
            else:
                lines.append(f"| {g} | {v['n']} | {v['pre']:.0f}% | {v['post']:.0f}% | {v['gain_pp']} |")
        if s["feynman"] and s["lecture"]:
            diff = round(s["feynman"]["gain_pp"] - s["lecture"]["gain_pp"], 1)
            lines.append("")
            lines.append(f"**组间提升差：{diff:+} pp**（feynman - lecture）")
        lines.append("")

    lines += ["## 明细", "",
              "| 用户 | 组 | 前测 | 后测 | 提升(pp) | 标记 |",
              "|---|---|---|---|---|---|"]
    for r in records:
        f = flag_map[r["user_id"]]
        pre_s = "-" if r["pre"] is None else f"{r['pre']*100:.0f}%"
        gain_s = "-" if r["gain_pp"] is None else str(r["gain_pp"])
        lines.append(
            f"| {r['user_id']} | {r['group']} | {pre_s} | {r['post']*100:.0f}% | "
            f"{gain_s} | {','.join(f) or ''} |")

    # E1 回流统计（次要指标：回流是闭环验证，不参与组间对照）
    rf = analysis["reflow"]
    total_rf = rf["completed"] + rf["failed"] + rf["given_up"] + rf["open"]
    gains = rf["retest_gains"]
    gains_s = (f"{sum(gains)/len(gains)*100:+.1f}pp" if gains else "-")
    lines += ["", "## 回流统计（E1 Loop 工程化，PLAN 20.5）", "",
              f"- 回流任务：{total_rf} 个（完成 {rf['completed']} / 续轮 {rf['failed']} / "
              f"超轮放弃 {rf['given_up']} / 进行中 {rf['open']}）",
              f"- 重测较触发时提升均值：{gains_s}（n={len(gains)}）",
              f"- 注：组间对照取**第一条**后测，回流重测不计入（防刷分污染）"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="C 阶段实验分析")
    ap.add_argument("--db", default="data/state.db",
                    help="实验库路径（默认 data/state.db，容器卷挂载库）")
    ap.add_argument("--out", default="reports/experiment.md", help="报告输出路径")
    args = ap.parse_args()

    analysis = analyze(args.db)
    report = render(analysis)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[报告已写入 {out}]")


if __name__ == "__main__":
    main()
