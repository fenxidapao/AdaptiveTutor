"""
eval_local.py — 本地自实现 RAG 评估（W3 兜底方案）

为什么不用 ragas？
  ragas 0.1.22 / 0.2.x 在 import 阶段会拉取 langchain_community.chat_models.vertexai，
  而 langchain_community 0.3.x 已移除该符号，无法共存（pip 还会把 0.1.22 自动升回 0.2.x）。
  故放弃 ragas，改成 LLM-as-judge，自己实现四个核心指标：
    1. faithfulness     忠实度：回答是否只由检索到的上下文支撑（防幻觉）
    2. answer_relevancy 答案相关性：回答是否切题、直接回应问题
    3. context_precision 上下文精度：相关段落是否排在了召回结果的前面
    4. context_recall    上下文召回：标准答案里的事实是否都被检索到的上下文覆盖

特点：
  - 零额外依赖（不装 ragas / datasets / nest_asyncio）；裁判默认独立模型 qwen2.5:3b
    （T2：与生成模型 llama3.1:8b 不同源，破除「同模型既生成又打分」的自评偏差；
    可切 JUDGE_PROVIDER=deepseek 走 API，见 config.py）
  - 四个指标 = 你自己的代码，自实现评估
  - 复用 rag_core.retrieve_and_rerank 与 build_rag_chain，保证『评估用的来源』=『线上用的来源』
  - 输出 eval_results.csv + 控制台汇总

运行：
  python eval_local.py
"""
import json
import re
import csv
import os
import urllib.request

import config
from rag_core import (
    build_rag_chain,
    retrieve_and_rerank,
    load_and_chunk_documents,
    get_vector_store,
    ollama_chat,
)

# ====================== T2: 独立裁判（2026-08-22）======================
# 原实现裁判复用 rag_core.ollama_chat（与生成同模型 llama3.1:8b），存在自评价偏差。
# 现按 config.JUDGE_PROVIDER 分流：
#   local    → Ollama 本地独立模型（默认 qwen2.5:3b），数据不出本机
#   deepseek → DeepSeek API（需 .env 配 JUDGE_API_KEY），效果更强但数据出本机
import ollama
_judge_client = ollama.Client(host=config.OLLAMA_BASE_URL)


def _judge_local(messages):
    """本地 Ollama 裁判（独立模型，温度 0 逼稳定 JSON）。"""
    resp = _judge_client.chat(
        model=config.JUDGE_MODEL, messages=messages, options={"temperature": 0.0}
    )
    return resp["message"]["content"]


def _judge_deepseek(messages):
    """DeepSeek API 裁判（OpenAI 兼容 /chat/completions，标准库实现零新依赖）。
    模型固定 deepseek-chat；JUDGE_MODEL 是 Ollama 的模型名，两种 provider 各自独立。"""
    if not config.JUDGE_API_KEY:
        raise RuntimeError("JUDGE_PROVIDER=deepseek 但 .env 未设置 JUDGE_API_KEY")
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        config.JUDGE_API_BASE + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.JUDGE_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]

# 裁判模型：温度设 0，逼它稳定输出可解析的 JSON（T2 起默认独立模型 qwen2.5:3b，可切 DeepSeek API）

EVAL_CSV = os.path.join(
    os.path.dirname(__file__), os.environ.get("EVAL_CSV_NAME", "eval_results.csv")
)

# 与 summarize_and_save / 增量落盘共用的字段顺序（模块级，避免重复定义）
FIELDS = [
    "question",
    "answer",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "faithfulness_detail",
    "context_precision_detail",
    "context_recall_detail",
]


# ====================== JSON 解析兜底 ======================
# T2 补充：小裁判（qwen2.5:3b）偶发把「任务指令」当成回答陈述拆出（人工核对发现的
# 解析故障，judge_review_20.csv 第 13-14/33-35/53-55 行等），这里做二次清洗。
_STMT_NOISE_PREFIX = (
    "把【回答】拆成",
    "把【标准答案】拆成",
    "对每条陈述",
    "对每条事实",
    "计算 score",
    "只输出 JSON",
    "任务：",
)


def _clean_judgment_items(items, key="statement"):
    """过滤被误拆进 statements/facts 的任务指令类条目。"""
    out = []
    for it in items or []:
        text = (it.get(key) or "").strip()
        if not text:
            continue
        if text.startswith(_STMT_NOISE_PREFIX):
            continue
        out.append(it)
    return out


def _repair_judgment_items(text: str):
    """P2 复核新增：JSON 解析失败的容错提取。

    根因（实测）：裁判在陈述文本里嵌入未转义双引号（如代码 type=["csv","json"]）
    时 json.loads 必然失败，且温度 0 下重试输出几乎相同 → 重试耗尽返回 {}，
    该题静默记 0 分（冻结测试集 Q11/Q17 空 detail 即此因）。
    做法：按 "statement"/"fact" 锚点把残缺 JSON 切成条目段，段内贪心匹配到
    supported/covered 键——段内含任意引号均可正确提取。
    """
    keys = []
    for key, flag in (("statement", "supported"), ("fact", "covered")):
        if f'"{key}"' in text:
            keys.append((key, flag))
    for key, flag in keys:
        starts = [m.start() for m in re.finditer(rf'"{key}"\s*:', text)]
        items = []
        for i, s in enumerate(starts):
            seg = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)]
            m = re.search(
                rf'"{key}"\s*:\s*"(.*)"\s*,\s*"{flag}"\s*:\s*(true|false)',
                seg,
                re.DOTALL,
            )
            if m:
                items.append({key: m.group(1), flag: m.group(2) == "true"})
        if items:
            sup = sum(1 for it in items if it[flag])
            return {f"{key}s": items, "score": round(sup / len(items), 3)}
    return None


def _extract_json(text: str):
    """从模型输出里尽量抠出第一个合法 JSON 对象；兼容 ```json 围栏和前后废话。
    解析失败时降级到 _repair_judgment_items 容错提取（见其 docstring）。"""
    if not text:
        return None
    # 去 markdown 围栏
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # 抠第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _repair_judgment_items(text)
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return _repair_judgment_items(blob)


def _judge_once(system: str, user: str, retries: int = 2):
    """让裁判模型回答一次，返回解析后的 dict；失败重试，最终仍失败返回 {}。
    T2（2026-08-22）：裁判按 config.JUDGE_PROVIDER 选择——local=Ollama 独立模型
    （默认 qwen2.5:3b，与生成模型不同源）；deepseek=DeepSeek API。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(retries + 1):
        try:
            if config.JUDGE_PROVIDER == "deepseek":
                content = _judge_deepseek(messages)
            else:
                content = _judge_local(messages)
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
        except Exception as e:  # 网络/超时等，下一轮重试
            last_err = e
    return {}


# ====================== 四个指标 ======================
def _content_sentence_estimate(answer: str) -> int:
    """估计回答的正文句数（代码块/行内代码整体各算 1 句），用作陈述覆盖下限。

    Q15 提取塌缩（2026-08-31）根因之一：裁判把多句回答只拆出 1 条陈述，
    漏掉的句子恰好是被支撑的支撑句 → 单句误判把 faith 打成 0。
    这里用保守切分（。！？；换行）估句数，仅触发重试，不参与计分。
    """
    text = answer or ""
    est = len(re.findall(r"```.*?```", text, flags=re.DOTALL))
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    est += sum(1 for seg in re.split(r"[。！？；\n]+", text) if seg.strip())
    return max(est, 1)


def faithfulness(context: str, answer: str) -> dict:
    """
    忠实度：把回答拆成若干独立陈述，逐条判断是否可由上下文推导。
    score = 被支撑的陈述数 / 总陈述数，范围 0~1。
    """
    system = (
        "你是严谨的 RAG 评估裁判。只依据给定的【上下文】判断，不要使用外部知识。"
        "必须只输出一个 JSON 对象，不要任何额外文字或解释。"
    )
    user = f"""【上下文】
{context}

【回答】
{answer}

任务：
1. 只把【回答】的正文内容拆成若干条彼此独立的陈述（statements）。
   注意：必须逐句覆盖——回答正文的每一句都要对应恰好一条陈述，不得合并、不得遗漏；
   例外：回答中的代码示例（代码块、命令行、示例片段）必须整体合并为一条陈述，
   不要逐行拆分；该陈述判断的是「示例用法是否与上下文一致」——核心参数/调用方式
   一致即为 supported，界面文案、变量命名等无关细节差异不影响判定。
2. 对每条陈述，判断是否能从【上下文】中直接推导或支持（supported 为 true/false）。
   注意：概括性重述（如「通过 X 可以实现 Y」类总结句）只要其核心主张能从上下文
   推导即为 supported——上下文允许转述，不要求措辞逐字对应；
   只有上下文推导不出的主张才判 false。
3. 计算 score = 被支撑的陈述数 / 总陈述数（0~1，保留两位小数）。

只输出 JSON，格式：
{{"statements": [{{"statement": "陈述内容", "supported": true}}], "score": 0.0}}"""
    out = _judge_once(system, user)
    items = _clean_judgment_items(out.get("statements", []))
    # 覆盖下限保护（Q15 提取塌缩修复）：陈述数 < 正文句数估计 → 裁判漏拆，
    # 带强化指令重试一次，取覆盖更全的一份。漏拆会让被漏句的支撑与否完全随机
    # 影响分数（Q15 三轮 0/0.5/0 的直接根因）。仅多花一次裁判调用，命中即重试。
    est = _content_sentence_estimate(answer)
    if items and len(items) < est:
        user_retry = (
            user
            + f"\n\n注意：上一轮只拆出 {len(items)} 条陈述，但回答正文约有 {est} 句；"
            "必须逐句覆盖——每一句正文对应一条陈述，不得合并或遗漏（代码示例整体一条除外）。"
        )
        out_retry = _judge_once(system, user_retry)
        items_retry = _clean_judgment_items(out_retry.get("statements", []))
        if len(items_retry) > len(items):
            items = items_retry
    if items:
        # P2 复核：裁判自报 score 偶发与明细不一致（漏列陈述时出现 score=0.6 但
        # 明细 2/2 全 true）。统一以明细重算，保证每个分数可复算、可审计。
        sup = sum(1 for it in items if it.get("supported") is True)
        score = max(0.0, min(1.0, sup / len(items)))
        return {"score": round(score, 3), "detail": items}
    # 明细为空（裁判解析失败/未拆出任何陈述）：如实标注而非静默记 0 分，
    # 让下游能看到「这是裁判故障」而不是「答案完全不忠实」。
    try:
        fallback = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        fallback = 0.0
    return {
        "score": round(max(0.0, min(1.0, fallback)), 3),
        "detail": [{"statement": "JUDGE_FAILED: 裁判未返回有效拆解", "supported": False}],
    }


def answer_relevancy(question: str, answer: str) -> dict:
    """
    答案相关性：回答是否切题、是否直接回应问题、是否大量冗余。
    score 0=答非所问，1=完全切题且充分。
    """
    system = (
        "你是 RAG 评估裁判。判断回答是否切题。只输出一个 JSON 对象，不要额外文字。"
    )
    user = f"""【问题】
{question}

【回答】
{answer}

任务：
判断回答是否切题、是否直接回应了问题、是否包含大量与问题无关的冗余内容。
score 取值 0~1（保留两位小数）：1 表示完全切题且充分回应，0 表示答非所问。

只输出 JSON，格式：
{{"score": 0.0, "reason": "简短理由"}}"""
    out = _judge_once(system, user)
    score = float(out.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return {"score": round(score, 3), "detail": out.get("reason", "")}


def context_precision(question: str, docs) -> dict:
    """
    上下文精度：检索召回的段落是否『相关的排前面』。
    先把带序号的上下文交裁判逐条判相关（relevant），再按 RAGAS 公式算 precision。
    context_precision = Σ_k (precision@k × rel_k) / (相关总数)
    """
    numbered = "\n\n".join(
        f"[#{i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
    )
    system = (
        "你是 RAG 评估裁判。只判断上下文是否与问题相关，不要使用外部知识。"
        "必须只输出一个 JSON 对象，不要额外文字。"
    )
    user = f"""【问题】
{question}

【检索到的上下文（按相关性从高到低排列，已编号）】
{numbered}

任务：
对每条上下文（#[序号]）判断是否与【问题】相关（relevant 为 true/false）。
只输出 JSON，格式：
{{"judgments": [{{"index": 1, "relevant": true}}, {{"index": 2, "relevant": false}}]}}"""
    out = _judge_once(system, user)
    judgments = out.get("judgments", [])
    if not judgments:
        return {"score": 0.0, "detail": "裁判未返回有效判定"}

    # 按 index 升序还原顺序
    rel = [0] * len(docs)
    for j in judgments:
        idx = j.get("index")
        if isinstance(idx, int) and 1 <= idx <= len(docs):
            rel[idx - 1] = 1 if j.get("relevant") else 0

    num_rel = sum(rel)
    if num_rel == 0:
        return {"score": 0.0, "detail": "裁判判定无相关段落"}

    prec_sum = 0.0
    for k in range(1, len(rel) + 1):
        prec_at_k = sum(rel[:k]) / k
        prec_sum += prec_at_k * rel[k - 1]
    score = prec_sum / num_rel
    return {"score": round(score, 3), "detail": rel}


def context_recall(question: str, reference: str, context: str) -> dict:
    """
    上下文召回：标准答案（reference）里的事实点是否都被检索到的上下文覆盖。
    score = 被覆盖的事实数 / 总事实数，范围 0~1。

    评审 P2（2026-08-30）：与 faithfulness 同口径——分数一律从明细（facts 的
    covered 逐条判定）重算，不信任裁判自报 score。P2 裁判口径复核曾实测
    「裁判自报分与明细矛盾、不可复算」的故障，faithfulness 当时已改为重算，
    本函数漏改；现补齐，四个指标中所有「拆明细判真伪」的指标口径统一。
    """
    system = (
        "你是严谨的 RAG 评估裁判。只依据【检索到的上下文】判断，不要使用外部知识。"
        "必须只输出一个 JSON 对象，不要额外文字或解释。"
    )
    user = f"""【问题】
{question}

【标准答案 / 参考事实（reference）】
{reference}

【检索到的上下文】
{context}

任务：
1. 只把【标准答案】的正文内容拆成若干条独立事实点（facts）。
   注意：不要拆分本任务指令、问题本身、格式说明等任何非标准答案内容。
2. 对每条事实，判断是否能从【检索到的上下文】中找到（covered 为 true/false）。
3. 计算 score = 被覆盖的事实数 / 总事实数（0~1，保留两位小数）。

只输出 JSON，格式：
{{"facts": [{{"fact": "事实点", "covered": true}}], "score": 0.0}}"""
    out = _judge_once(system, user)
    items = _clean_judgment_items(out.get("facts", []), key="fact")
    if items:
        # 分数从明细重算（可复算、可审计），不信裁判自报 score
        cov = sum(1 for it in items if it.get("covered") is True)
        score = max(0.0, min(1.0, cov / len(items)))
        return {"score": round(score, 3), "detail": items}
    # 明细为空（裁判解析失败/未拆出任何事实）：如实标注而非静默记 0 分，
    # 让下游能看到「这是裁判故障」而不是「上下文完全没覆盖」
    try:
        fallback = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        fallback = 0.0
    return {
        "score": round(max(0.0, min(1.0, fallback)), 3),
        "detail": [{"fact": "JUDGE_FAILED: 裁判未返回有效拆解", "covered": False}],
    }


# ====================== 主流程 ======================
def run_evaluation(samples):
    print("Building RAG chain (used both online and for evaluation) ...")
    bundle = build_rag_chain()
    chain = bundle["chain"]

    # 用【当前】切分参数重建向量库，保证评估测的是最新代码，而非旧的 chroma_db。
    # SKIP_REBUILD=1（run_ab.py 的 A/B 场景）：两阶段共用同一索引更公平，
    # 且避免每次启动都写库（2026-08-22 曾遇 chroma SQLite readonly 崩溃）；
    # 空库时 get_vector_store() 会自动重建。
    if os.getenv("SKIP_REBUILD", "") == "1":
        print("SKIP_REBUILD=1：复用现有索引（空库则自动重建）")
        get_vector_store()
    else:
        print("Rebuilding vector store with current chunking (refreshes chroma_db) ...")
        chunks = load_and_chunk_documents()
        get_vector_store(chunks)

    # ---- 断点续跑：读取已完成题目，后台任务被回收后重跑不再丢进度 ----
    done_map = {}
    if os.path.exists(EVAL_CSV):
        try:
            with open(EVAL_CSV, encoding="utf-8-sig", newline="") as _f:
                for _row in csv.DictReader(_f):
                    # CSV 读回的值都是字符串，续跑塞进 results 后会被求和，
                    # 必须把数值字段转回 float，否则 summarize 时 int+str 崩溃。
                    for _k in (
                        "faithfulness",
                        "answer_relevancy",
                        "context_precision",
                        "context_recall",
                    ):
                        try:
                            _row[_k] = float(_row.get(_k, 0.0) or 0.0)
                        except (TypeError, ValueError):
                            _row[_k] = 0.0
                    done_map[_row["question"]] = _row
            if done_map:
                print(f"检测到已完成 {len(done_map)} 题，将从断点续跑。")
        except Exception as _e:
            print(f"[warn] 读取已有结果失败，将从头跑: {_e!r}")
            done_map = {}

    _write_header = not os.path.exists(EVAL_CSV)
    _fout = open(EVAL_CSV, "a", encoding="utf-8-sig", newline="")
    _w = csv.DictWriter(_fout, fieldnames=FIELDS)
    if _write_header:
        _w.writeheader()
        _fout.flush()

    results = []
    for i, s in enumerate(samples, 1):
        q = s["question"]
        ref = s.get("reference", "")
        if q in done_map:
            print(f"\n[{i}/{len(samples)}] Q: {q}  [跳过-已完成]")
            results.append(done_map[q])
            continue
        print(f"\n[{i}/{len(samples)}] Q: {q}")

        # 单题容错：任一题处理失败（Ollama 抖动 / 解析异常）记为 0 分并跳过，
        # 不让整轮评估因一题崩溃而白跑。
        try:
            # 步骤1：检索（UI 与链同一条入口）
            docs = retrieve_and_rerank(q)
            context = "\n\n".join(d.page_content for d in docs)

            # 步骤2：生成回答（与线上一致）
            answer = chain.invoke(q)
            print(f"    A: {answer[:80].replace(chr(10), ' ')}...")

            # 步骤3：四个指标（本机 Ollama 当裁判）
            fth = faithfulness(context, answer)
            arl = answer_relevancy(q, answer)
            cprec = context_precision(q, docs)
            crec = context_recall(q, ref, context)
            print(
                f"    faithfulness={fth['score']}  answer_relevancy={arl['score']}  "
                f"context_precision={cprec['score']}  context_recall={crec['score']}"
            )

            row = {
                "question": q,
                "answer": answer.replace("\n", " ").strip(),
                "faithfulness": fth["score"],
                "answer_relevancy": arl["score"],
                "context_precision": cprec["score"],
                "context_recall": crec["score"],
                "faithfulness_detail": json.dumps(fth["detail"], ensure_ascii=False),
                "context_precision_detail": json.dumps(cprec["detail"], ensure_ascii=False),
                "context_recall_detail": json.dumps(crec["detail"], ensure_ascii=False),
            }
        except Exception as e:
            print(f"    [ERROR] 该题处理失败，记 0 分跳过: {e!r}")
            row = {
                "question": q,
                "answer": f"ERROR: {e!r}",
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness_detail": "",
                "context_precision_detail": "",
                "context_recall_detail": "",
            }
        # 逐题增量落盘（flush），被回收也不丢已完成进度
        _w.writerow(row)
        _fout.flush()
        results.append(row)

    _fout.close()
    return results


def summarize_and_save(results):
    n = len(results) or 1
    avg = {
        k: round(sum(r[k] for r in results) / n, 3)
        for k in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
    }
    print("\n===== 评估结果汇总（本机 Ollama 裁判）=====")
    print(f"样本数: {len(results)}")
    for k, v in avg.items():
        print(f"  {k:18s}: {v}")

    # 写入 CSV（含逐条明细，方便复盘哪些题拉分）
    fields = FIELDS
    import tempfile, os

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", prefix=".eval_tmp_", dir=os.path.dirname(os.path.abspath(EVAL_CSV))
    )
    try:
        with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        # 原子替换：先写临时文件再 rename 覆盖，避免瞬时锁导致整轮 2 小时白跑
        last_err = None
        for attempt in range(1, 6):
            try:
                os.replace(tmp_path, EVAL_CSV)
                break
            except PermissionError as e:
                last_err = e
                import time
                time.sleep(2 * attempt)
        else:
            raise last_err  # 5 次重试仍失败才真的抛
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    print(f"\n明细已写入: {EVAL_CSV}")
    return avg


if __name__ == "__main__":
    from eval_data import EVAL_SAMPLES

    res = run_evaluation(EVAL_SAMPLES)
    summarize_and_save(res)
