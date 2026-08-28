"""知识检索封装：HTTP 调 CourseRAG + LLM 内容相关性过滤 + notes 兜底。

检索策略（2026-08-28 对齐契约）：
  CourseRAG /retrieve 的 RetrieveRequest 仅认 question/mode（不接受 top_k/sources），
  实际返回条数由 CourseRAG 自己的 RERANKER_TOP_N 决定（本端只做截断保险）。
  采用 accurate 模式（向量+BM25 融合/reranker），返回带 score，便于按相关度加权。
  流程：检索 top-N → LLM 按"内容是否与知识点相关"过滤 → 空则 notes 兜底。
  零侵入 CourseRAG（三项目独立原则），不依赖 source 文件名。

对外接口：
  retrieve_tutor_context(kp) -> {"context": str, "source": "rag"|"notes", "used_docs": [...]}
"""

import json
import urllib.error
import urllib.request

import config
import learning_pack
import model
import prompts

# 讲解与"相关性过滤"合并为一次 LLM 调用（prompt 已版本化：prompts/rag_filter.md，D3）


class RAGError(Exception):
    """CourseRAG 不可用。"""


def _retrieve_raw(question: str, top_n: int | None = None) -> list[dict]:
    """调 CourseRAG /retrieve，返回 docs 列表（带 source/content/score 字段）。

    注意：CourseRAG 不接受 top_k，实际条数以它自身 RERANKER_TOP_N 为准；
    这里 top_n 仅作客户端再保险（截断，与上游取较小）。
    """
    top_n = top_n or config.RAG_RETRIEVE_TOP_N
    payload = json.dumps({
        "question": question,
        "mode": config.RAG_MODE,
    }).encode("utf-8")
    req = urllib.request.Request(
        config.RAG_BASE_URL + "/retrieve",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        raise RAGError(f"CourseRAG 不可用: {e}") from e
    return body.get("docs", [])[: top_n]


def _llm_filter(kp: dict, docs: list[dict]) -> tuple[list[int], str]:
    """LLM 按内容相关性过滤 + 组织讲解。返回 (used_indexes, explanation)。"""
    lines = []
    for i, d in enumerate(docs):
        score = ""
        if d.get("score") is not None:
            try:
                score = f", 相关度:{float(d['score']):.3f}"
            except (TypeError, ValueError):
                score = ""
        snippet = str(d.get("content", ""))[: config.RAG_DOC_CHAR_LIMIT]
        lines.append(f"[{i}] (来源:{d.get('source', '?')}{score})\n{snippet}")
    doc_text = "\n".join(lines)
    prompt = prompts.load("rag_filter.md").format(
        kp_title=kp.get("title", kp["kp_id"]),
        kp_id=kp["kp_id"],
        docs=doc_text,
    )
    raw = model.chat(
        [{"role": "system", "content": "你只输出合法 JSON，不加任何解释。"},
         {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=1600, caller="rag_filter",
    )
    # 容错：提取 JSON 子串（模型可能带 ```json 围栏）
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        used = [int(x) for x in data.get("used", [])]
        explanation = data.get("explanation", "").strip()
        return used, explanation
    except (json.JSONDecodeError, ValueError):
        return [], ""


def retrieve_tutor_context(kp_id: str, course: str = "python") -> dict:
    """讲解材料：RAG 检索 → LLM 过滤 → notes 兜底。

    返回 {"context": 讲解材料文本, "source": "rag"|"notes", "used_sources": [...]}
    """
    graph = learning_pack.load_graph(course)
    kp = graph["_by_id"].get(kp_id)
    if kp is None:
        raise learning_pack.LearningPackError(f"未知知识点: {kp_id}")

    # 1) RAG 检索
    try:
        docs = _retrieve_raw(kp.get("title", kp_id))
    except RAGError as e:
        print(f"[i] {e} → 使用 notes 兜底")
        docs = []

    if docs:
        used, explanation = _llm_filter(kp, docs)
        if used:
            used_sources = sorted({docs[i].get("source", "?") for i in used})
            return {
                "context": explanation,
                "source": "rag",
                "used_sources": used_sources,
            }

    # 2) notes 兜底
    notes = learning_pack.notes_for(course, kp_id)
    if notes:
        return {"context": notes, "source": "notes", "used_sources": ["notes/"]}

    return {"context": "", "source": "none", "used_sources": []}
