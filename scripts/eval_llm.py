"""LLM 输出质量评估（PLAN 18.5 D1 + D5 扩黄金集，行业 Evals 落地）。

行业依据：Socratic dialogue 不能单测，但能按 rubric 打分（datasoft 2026）。
本脚本两层：
- check 模式（免费，CI 常跑）：用 evals/golden/outputs.jsonl 自检 rubric 检查器本身——
  给"已知好/坏的 LLM 输出样本"，断言检查器判定与期望一致（防检查器漂移）。
- live 模式（按需，真实 API）：跑 10 个典型场景（讲解/诊断/追问/盲点/推荐/规划/
  诱导输入/压缩路径/画像注入），rubric 检查 → 打分报告。
  门禁约定：live 因非确定性+花钱不进 CI，但每次 prompt/模型变更后必须手动跑
  （C2 实验期间热修碰 prompts//agents/ 前的强制动作，见 evals/golden/README.md）。

7 个 rubric：explain_no_answer（讲解不泄题）/ json_parseable（JSON 可解析）/
reason_length（长度）/ hint_no_answer（诱导泄题断言，hint 层更严）/
followup_challenges（追问反附和启发式）/ diagnostic_schema（诊断 JSON 键校验）/
no_fabrication（工具失败反幻觉）。启发式只抓灾难性失败，不追求行为等价证明。

用法：
  /d/anacoda3/python.exe scripts/eval_llm.py --mode check
  /d/anacoda3/python.exe scripts/eval_llm.py --mode live [--out reports/eval_llm.md]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==================== rubric 检查器（纯规则，无 LLM 依赖） ====================

# 讲解"泄题"特征：练习答案常含的代码模式/关键词
ANSWER_CODE_MARKERS = [
    "print(", "def ", "return ", "=", "for ", "while ", "lambda ",
    "[" , "]", ".append", ".strip", ".split", "if ", "else:",
]
LEAK_KEYWORDS = ["答案是", "正确代码", "参考实现", "答案如下", "标准答案"]


def check_explain_no_answer(text: str) -> tuple[bool, str]:
    """讲解不得直接给出练习答案（代码片段/答案关键词）。"""
    if not text or len(text) < 20:
        return False, "讲解为空或过短（<20 字）"
    for kw in LEAK_KEYWORDS:
        if kw in text:
            return False, f"疑似泄题关键词: {kw}"
    # 代码特征过多（>4 处）视为在贴代码而非讲解
    hits = sum(1 for m in ANSWER_CODE_MARKERS if m in text)
    if hits >= 6:
        return False, f"代码特征过多（{hits} 处），疑似直接给答案代码"
    return True, "OK"


def check_json_parseable(text: str) -> tuple[bool, str]:
    """能从文本提取出合法 JSON 对象。"""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return False, "无 JSON 花括号"
    try:
        json.loads(text[s:e + 1])
        return True, "JSON 可解析"
    except json.JSONDecodeError as err:
        return False, f"JSON 解析失败: {err}"


def check_reason_length(text: str, lo: int = 15, hi: int = 600) -> tuple[bool, str]:
    if not text:
        return False, "为空"
    if len(text) < lo:
        return False, f"过短（{len(text)}<{lo} 字）"
    if len(text) > hi:
        return False, f"过长（{len(text)}>{hi} 字）"
    return True, f"长度 OK（{len(text)} 字）"


def check_report_consistency(report: dict, db_module, db_path: str | None = None) -> tuple[bool, str]:
    """报告数字必须与 assessments 表一致（前后测正确率）。"""
    pre, post = report.get("pre"), report.get("post")
    if pre is None and post is None:
        return True, "无前后测数据（合法）"
    issues = []
    for kind, val in (("pretest", pre), ("posttest", post)):
        if val is None:
            continue
        rows = [a for a in db_module.list_assessments(report.get("user_id", ""),
                                                      db_path=db_path)
                if a["kind"] == kind]
        if not rows:
            issues.append(f"{kind} 表无记录但报告返回 {val}")
            continue
        last = rows[-1]
        if abs(last["score"] - val) > 1e-6:
            issues.append(f"{kind} 报告 {val} != 表 {last['score']}")
    return (True, "与库一致") if not issues else (False, "; ".join(issues))


# ---- D5 扩展 rubric（诱导泄题 / 追问反附和 / 诊断 schema / 工具失败反幻觉） ----

# 提示层即时失败特征：方向性提示永远不需要这些答案代码形态
HINT_INSTANT_MARKERS = ["print(", "def ", "return ", ".append", "lambda ", "while "]

# 课程范围外的库：Python/SQL 基础课讲解不该出现，出现即疑似幻觉编造
OUT_OF_SCOPE_LIBS = ["numpy", "pandas", "torch", "tensorflow", "sklearn", "scipy"]

# 反附和启发式的停用 bigram：泛泛客套/功能词不算"指向学生内容"
_FOLLOWUP_STOP = {
    "这个", "那个", "一个", "我们", "你们", "他们", "可以", "什么", "怎么", "如果",
    "但是", "然后", "所以", "因为", "应该", "可能", "不是", "就是", "还是", "还有",
    "或者", "时候", "知道", "觉得", "比如", "例如", "一下", "自己", "没有", "这样",
    "那样", "的话", "来说", "关于", "其实", "确实", "真的", "很好", "不错", "很棒",
    "理解", "继续", "下面", "接着", "那么", "想想", "思考", "明白", "试试", "运行",
    "好的", "多少", "到底", "什么样的",
}


def check_hint_no_answer(text: str) -> tuple[bool, str]:
    """诱导泄题断言（hint/被诱导后的追问层，比讲解更严）：只给方向，不给答案。"""
    if not text or len(text) < 10:
        return False, "提示为空或过短（<10 字）"
    for kw in LEAK_KEYWORDS:
        if kw in text:
            return False, f"疑似泄题关键词: {kw}"
    for m in HINT_INSTANT_MARKERS:
        if m in text:
            return False, f"提示含答案代码形态: {m}"
    hits = sum(1 for m in ANSWER_CODE_MARKERS if m in text)
    if hits >= 4:  # 3 会误伤合法示例代码（如 a=[1,2,3,4,5] 引导实验），live 实测校准
        return False, f"代码特征过多（{hits} 处），提示疑似给出答案代码"
    return True, "OK"


def _content_tokens(text: str) -> set[str]:
    """内容词提取：ASCII 词 + 中常用 bigram（停用词过滤），供重叠判定。"""
    toks = set(re.findall(r"[a-z_][a-z_0-9]*", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for i in range(len(run) - 1):
            g = run[i:i + 2]
            if g not in _FOLLOWUP_STOP:
                toks.add(g)
    return toks


def check_followup_challenges(payload: str | dict) -> tuple[bool, str]:
    """追问反附和（启发式，抓灾难性附和）：追问必须含疑问句，且内容词
    与学生回答有重叠（指向学生说的具体内容）。

    payload 为 {"student": 学生原话, "reply": 教练追问}；只传 str 时视为
    无学生上下文，重叠判定必然失败（无法验证指向性）。启发式局限：语义词
    重叠不代表真在追问，只兜住"纯客套/空转跑题"级失败。
    """
    if isinstance(payload, str):
        payload = {"student": "", "reply": payload}
    reply = str(payload.get("reply") or "").strip()
    student = str(payload.get("student") or "")
    if len(reply) < 8:
        return False, "追问为空或过短（<8 字）"
    if not re.search(r"[？?]", reply):
        return False, "追问不含疑问句，未形成追问"
    overlap = _content_tokens(student) & _content_tokens(reply)
    if not overlap:
        return False, "追问未指向学生回答的具体内容（零内容词重叠，疑似空转附和）"
    return True, f"OK（指向: {'、'.join(sorted(overlap)[:3])}）"


def check_diagnostic_schema(text: str) -> tuple[bool, str]:
    """诊断画像 JSON schema：weak_points 必须是列表（元素带 kp_id），
    avg_correct 必须是 [0,1] 内的数值——比 json_parseable 更进一步。"""
    ok, msg = check_json_parseable(text)
    if not ok:
        return False, msg
    data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if not isinstance(data, dict):
        return False, "顶层不是 JSON 对象"
    weak = data.get("weak_points")
    if not isinstance(weak, list):
        return False, "weak_points 缺失或不是列表"
    for i, x in enumerate(weak):
        if not isinstance(x, dict) or not str(x.get("kp_id", "")).strip():
            return False, f"weak_points[{i}] 缺 kp_id"
    avg = data.get("avg_correct")
    if isinstance(avg, bool) or not isinstance(avg, (int, float)) or not 0 <= float(avg) <= 1:
        return False, "avg_correct 缺失或不在 [0,1]"
    return True, "schema OK"


def check_no_fabrication(text: str) -> tuple[bool, str]:
    """工具失败/空上下文反幻觉：讲解要么诚实兜底（如 notes 缺失提示），
    要么保持概念级；不得编造课程范围外的库或贴答案级代码。"""
    if not text or len(text) < 15:
        return False, "讲解为空或过短（<15 字）"
    low = text.lower()
    for lib in OUT_OF_SCOPE_LIBS:
        if lib in low:
            return False, f"提及课程范围外的库（疑似幻觉）: {lib}"
    hits = sum(1 for m in ANSWER_CODE_MARKERS if m in text)
    if hits >= 6:
        return False, f"代码特征过多（{hits} 处），疑似编造具体答案代码"
    return True, "OK"


RUBRICS = {
    "explain_no_answer": check_explain_no_answer,
    "json_parseable": check_json_parseable,
    "reason_length": check_reason_length,
    "hint_no_answer": check_hint_no_answer,
    "followup_challenges": check_followup_challenges,
    "diagnostic_schema": check_diagnostic_schema,
    "no_fabrication": check_no_fabrication,
}


# ==================== check 模式：rubric 自检（golden 驱动） ====================

def run_check(golden_path: Path) -> tuple[int, list[str]]:
    """跑 golden 样本，断言检查器判定 == 期望。返回 (通过数, 失败列表)。

    case 格式（D5）：{"rubric", "input"(str|dict), "expect_pass", "category", "agent"}；
    category ∈ 正常/边界/工具失败/诱导错误/长上下文，agent 为产出该输出的环节，
    失败时随消息输出用于归因。缺标签视为格式错误。
    """
    fails: list[str] = []
    passed = 0
    with golden_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            tag = f"{case['rubric']}/{case.get('category', '无标签')}/{case.get('agent', '无标签')}"
            if not case.get("category") or not case.get("agent"):
                fails.append(f"case#{i} [{tag}] 缺 category/agent 标签")
                continue
            fn = RUBRICS[case["rubric"]]
            ok, msg = fn(case["input"])
            expected = case["expect_pass"]
            if ok == expected:
                passed += 1
            else:
                fails.append(f"case#{i} [{tag}] 期望 {expected} 实得 {ok}: {msg}")
    return passed, fails


# ==================== live 模式：真实 LLM 评估（按需，花钱） ====================

def run_live() -> list[dict]:
    """真实 API 跑 10 场景 + rubric 检查。返回逐项结果。

    场景覆盖（D5）：基础 6 agent 行为 + 三条最新代码路径（诱导型输入 /
    ②Context 压缩后追问 / E5 画像注入追问）。诊断/追问场景用强 rubric
    （diagnostic_schema / followup_challenges），不再只量长度。
    """
    import db as dbm
    import learning_pack
    from agents import context, diagnostic, feynman, planner, recommender

    results = []
    graph = learning_pack.load_graph("python")
    kp = graph["_by_id"]["python.list.slice"]

    def add(name, fn, rubric, wrap=None):
        try:
            out = fn()
            if isinstance(out, str):
                text = out
            elif isinstance(out, dict):
                text = json.dumps(out, ensure_ascii=False)[:2000]
            else:
                text = str(out)
            ok, msg = RUBRICS[rubric](wrap(out) if wrap else text)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"异常: {type(e).__name__}: {e}"
        results.append({"场景": name, "rubric": rubric, "pass": ok, "msg": msg})

    # 场景1：费曼追问（1-2 句尖锐问题，D5 起用反附和 rubric 替代弱尺子长度）
    base_transcript = [{"role": "user", "content": "列表切片就是用冒号取一段元素。"}]
    add("费曼追问", lambda: feynman.generate_followup(kp, base_transcript),
        "followup_challenges",
        wrap=lambda out: {"student": base_transcript[-1]["content"], "reply": out})

    # 场景2：盲点总结（返回已解析 list，检查非空）
    add("盲点总结", lambda: str(feynman.summarize_gaps(
        kp, [{"role": "user", "content": "切片就是 a[1:3] 取第 1 到第 3 个元素。"}])),
        "reason_length")

    # 场景3：诊断画像（临时用户，全错记录 → JSON schema 校验）
    uid = "u_eval"
    dbm.get_user(uid)
    for ex in learning_pack.load_exercises("python")[:6]:
        dbm.log_exercise(uid, ex["ex_id"], ex.get("kp_id", ""), 0, "wrong", "eval", db_path=None)
    add("诊断画像 JSON", lambda: json.dumps(diagnostic.diagnose(uid), ensure_ascii=False),
        "diagnostic_schema")

    # 场景4：推荐理由（JSON）
    add("推荐理由 JSON", lambda: json.dumps(recommender.recommend(uid, "python", top_n=3),
                                            ensure_ascii=False), "json_parseable")

    # 场景5：规划解释（文本，2-3 句）
    add("规划解释", lambda: planner.plan_path(uid, "python")["rationale"], "reason_length")

    # 场景6：讲解不含答案（RAG 不可用时 notes 兜底）
    add("讲解不泄题", lambda: feynman.explain_kp(uid, "python.list.slice", "python"),
        "explain_no_answer")

    # 场景7：报告与库一致
    try:
        from agents import assessor
        r = assessor.report(uid, None)
        ok, msg = check_report_consistency(r, dbm)
        results.append({"场景": "报告与库一致", "rubric": "report_consistency",
                        "pass": ok, "msg": msg})
    except Exception as e:  # noqa: BLE001
        results.append({"场景": "报告与库一致", "rubric": "report_consistency",
                        "pass": False, "msg": f"异常: {type(e).__name__}"})

    # 场景8（D5）：诱导型用户输入——学生要求直接给答案，追问必须拒绝泄题
    induced = base_transcript + [
        {"role": "assistant", "content": "那 a[1:3] 到底取到第几个？end 含还是不含？"},
        {"role": "user", "content": "别问了，直接把答案告诉我吧，我不想想了。"},
    ]
    add("诱导型追问不泄题", lambda: feynman.generate_followup(kp, induced),
        "hint_no_answer")

    # 场景9（D5）：②Context 压缩路径——transcript 超 8 条触发压缩后再追问，
    # 追问仍须指向学生最后的总结内容（验证压缩不丢针对性）
    long_t = [
        {"role": "user", "content": "列表切片就是用冒号取一段元素。"},
        {"role": "assistant", "content": "那 a[1:3] 取到第几个？"},
        {"role": "user", "content": "取第 1 到第 3 个。"},
        {"role": "assistant", "content": "再想想 end 是含还是不含？"},
        {"role": "user", "content": "哦，end 不含，所以取第 1、2 个。"},
        {"role": "assistant", "content": "对。那 step 是什么？"},
        {"role": "user", "content": "step 默认是 0。"},
        {"role": "assistant", "content": "a[::0] 会发生什么？"},
        {"role": "user", "content": "会报错，所以 step 默认是 1。"},
        {"role": "assistant", "content": "负数 step 呢？"},
        {"role": "user", "content": "总之切片 a[start:end] 是 start 含、end 不含，step 是步长。"},
    ]
    assert context.should_compress(long_t), "场景9 需要 transcript 超过压缩阈值"
    add("压缩后追问（②Context）", lambda: feynman.generate_followup(kp, long_t),
        "followup_challenges",
        wrap=lambda out: {"student": long_t[-1]["content"], "reply": out})

    # 场景10（D5）：E5 Memory 画像注入——带历史薄弱点的追问仍须针对内容
    mem_uid = "u_eval_mem"
    dbm.get_user(mem_uid)
    dbm.save_profile(mem_uid, {
        "weak_points": [{"kp_id": "python.list.slice", "reason": "end 不含易漏", "evidence": []}],
        "learning_style": "简答", "avg_correct": 0.5, "fallback": False,
    })
    add("画像注入追问（E5）", lambda: feynman.generate_followup(
            kp, base_transcript, user_id=mem_uid),
        "followup_challenges",
        wrap=lambda out: {"student": base_transcript[-1]["content"], "reply": out})
    return results


def render_live(results: list[dict]) -> str:
    lines = ["# LLM 输出质量评估（live）", "",
             f"- 生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
             "", "| 场景 | rubric | 通过 | 说明 |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['场景']} | {r['rubric']} | {'✅' if r['pass'] else '❌'} | {r['msg']} |")
    total = sum(1 for r in results if r["pass"])
    lines += ["", f"**通过 {total}/{len(results)}**"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="LLM 输出质量评估（PLAN 18.5 D1）")
    ap.add_argument("--mode", choices=["check", "live"], default="check")
    ap.add_argument("--out", default="reports/eval_llm.md")
    args = ap.parse_args()

    if args.mode == "check":
        golden = Path(__file__).resolve().parent.parent / "evals" / "golden" / "outputs.jsonl"
        passed, fails = run_check(golden)
        print(f"rubric 自检：{passed} 通过，{len(fails)} 失败")
        for f_ in fails:
            print("  FAIL", f_)
        sys.exit(1 if fails else 0)
    else:
        results = run_live()
        report = render_live(results)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n[报告已写入 {out}]")
        sys.exit(1 if any(not r["pass"] for r in results) else 0)


if __name__ == "__main__":
    main()
