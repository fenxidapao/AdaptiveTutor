"""u0 真实闭环冒烟（收尾用）：真实 DeepSeek LLM 跑通 ch1+ch2 完整闭环。

答案来自学习包正确解（mcq 用 check.answer 索引，output/code 用 SAMPLES），
费曼讲解喂固定文本模拟用户作答——本脚本是【系统链路验证】，非人类真实作答。

产出：assessments 表新增 ch1/ch2 的前后测记录（u0）；旧数据先备份到 state.db.bak_pre_v1
用法：/d/anacoda3/python.exe scripts/run_u0_smoke.py
"""
import builtins
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db  # noqa: E402
import learning_pack  # noqa: E402
import main as main_mod  # noqa: E402
from tests.verify_learning_pack import SAMPLES  # noqa: E402


def _mcq_answers() -> dict:
    ans = {}
    for q in learning_pack.load_pretest() + learning_pack.load_posttest():
        if q["type"] == "mcq":
            ans[q["ex_id"]] = str(q["check"]["answer"])
    for ex in learning_pack.load_exercises():
        if ex["type"] == "mcq":
            ans[ex["ex_id"]] = str(ex["check"]["answer"])
    return ans


def _build_feed(chapter: str) -> list[str]:
    ans = _mcq_answers()
    pre = [q for q in learning_pack.load_pretest() if q.get("chapter") == chapter]
    post = [q for q in learning_pack.load_posttest() if q.get("chapter") == chapter]
    kps = [k["kp_id"] for k in learning_pack.load_graph()["knowledge_points"]
           if k["chapter"] == chapter]
    ex_group = learning_pack.exercises_by_kp()

    feed = [ans[q["ex_id"]] for q in pre]          # 前测
    for kp in kps:                                  # 每 kp：费曼 3 轮 + 练习
        feed += ["变量就是给名字绑定值", "可以，比如切片就是取子集", "懂了"]
        for ex in ex_group.get(kp, [])[:3]:
            feed.append(ans.get(ex["ex_id"], "0") if ex["type"] == "mcq"
                        else SAMPLES.get(ex["ex_id"], "print(1)"))
    feed += [ans[q["ex_id"]] for q in post]         # 后测
    return feed


def main() -> None:
    # 备份并清理 u0 旧数据（留干净的本次前后测）
    if Path("state.db").exists():
        shutil.copy("state.db", "state.db.bak_pre_v1")
        print("[i] 旧 state.db 已备份: state.db.bak_pre_v1")
    with db._conn() as c:
        for t in ("assessments", "exercise_logs", "knowledge_points", "profile",
                  "reflow_logs"):
            c.execute(f"DELETE FROM {t} WHERE user_id='u0'")
    db.get_user("u0")

    for chapter in ("ch1", "ch2"):
        feed = _build_feed(chapter)
        it = iter(feed)
        builtins.input = lambda *a, **k: next(it)   # 喂答案（真实 LLM 不 mock）
        print(f"\n===== {chapter} 真实闭环开始（真实 DeepSeek API）=====")
        main_mod._learn_flow("u0", "python", "feynman", chapter)

    print("\n===== u0 前后测数据（assessments）=====")
    for a in db.get_assessments("u0"):
        print(f"  {a['kind']:9s} chapter={a['chapter']:4s} score={a['score']:.2f} "
              f"mode={a['mode']} total={a['total']}")


if __name__ == "__main__":
    main()
