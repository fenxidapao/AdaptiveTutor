"""
app.py — Streamlit 演示界面（改编自 RAG-KnowBot + W2 引用溯源增强）。
保留引用溯源（Sources 折叠面板），并升级为：带排名 / 相关性分数 / 原文偏移。

P0 改造（2026-08-18）：
  - P0-1 启动自检：侧边栏顶部状态条（依赖 / Ollama / Reranker 权重），消灭神秘转圈。
  - P0-2 对话记忆：最近 N 轮对话喂给 LLM；历史落盘 .chat_history.json，刷新不丢；可清空。
  - P0-3 检索体验：分阶段进度（向量召回 → [BM25 融合] → reranker 重排 → 生成）；「快速/精确」双模式
          （快速=纯向量 top-5 ~19s 默认；精确=向量+BM25 融合+reranker，救难 query）。

注意：P1-4 起删除文件只按 source 从索引中移除该文档（不再全量重建）；上传只增量 upsert 新文件块。
全量重建入口保留在侧边栏「全量重建索引（data/ 全部）」按钮（走 get_vector_store(chunks)）。
"""
import streamlit as st
import os
import json
import re

# reranker 权重（bge-reranker-v2-m3）加载需访问 HuggingFace。本机 huggingface.co 不可达会
# 超时卡死（页面一直转）。故在 import 任何可能拉起 sentence_transformers / huggingface_hub 的
# 模块之前设置——reranker.py 在首次调用时才 import sentence_transformers（间接 import huggingface_hub），
# 一旦它捕获到默认 HF_ENDPOINT=huggingface.co 就会卡死。
# 默认走镜像；若本机已缓存权重则强制离线，彻底不碰网络（权重已下载在 ~/.cache/huggingface）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_rerank_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
)
if os.path.isdir(_rerank_cache):
    os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from rag_core import (
    load_and_chunk_documents,
    get_vector_store,
    build_rag_chain,
    retrieve_and_rerank,
    chunk_file,
    add_documents,
    delete_by_source,
)
import config
from health import collect_status
from query_log import log_query

HISTORY_FILE = Path(config.DATA_DIR).parent / ".chat_history.json"
MAX_HISTORY_TURNS = 6  # 喂给 LLM 的最近轮数


# ==================== 对话历史持久化 ====================
def load_history():
    """启动时从磁盘恢复对话（刷新/重启不丢）。"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(messages):
    """每轮结束后落盘（仅存 role/content/citations，结构可 JSON 序列化）。"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_history_text(messages):
    """把历史拼成文本块喂给 LLM。"""
    turns = []
    for m in messages:
        role = "用户" if m["role"] == "user" else "助手"
        turns.append(f"{role}：{m['content']}")
    kept = turns[-MAX_HISTORY_TURNS * 2:] if turns else []
    return "\n".join(kept) if kept else "（无历史对话）"


def render_answer_html(answer, n_sources, mid):
    """P1-2 内联引用：把回答中的 [n]（含中文模型常用的全角【n】／［n］）渲染为可点击锚点
    （仅 1<=n<=n_sources 合法），非法编号原样保留。mid 为消息序号，避免多条消息锚点 id 冲突。"""
    # 半角 [n]、全角 【n】、全角 ［n］ 都映射到同一序号
    _BRACKET_RE = re.compile(r"\[(\d+)\]|【(\d+)】|［(\d+)］")

    def repl(m):
        k = int(m.group(1) or m.group(2) or m.group(3))
        if 1 <= k <= n_sources:
            return f'<sup><a href="#m{mid}-src-{k}" title="查看来源 {k}">[{k}]</a></sup>'
        return m.group(0)

    return _BRACKET_RE.sub(repl, answer)


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="CourseRAG - 专业课问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 会话状态 ====================
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = None
if "mode" not in st.session_state:
    st.session_state.mode = "fast"  # 默认快速模式（~19s），精确模式手动开启


# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📚 CourseRAG 控制台")

    # ---- P0-1 启动自检状态条 ----
    st.subheader("🔧 环境状态")
    for label, ok, msg in collect_status():
        if ok:
            st.success(f"✓ {label}：{msg}")
        else:
            st.error(f"✗ {label}：{msg}")
    st.caption("出问题先看这里，别对着转圈发呆。")

    st.markdown("---")

    # ---- P0-3 检索模式（P1-1：精确模式叠加 向量+BM25 混合检索）----
    st.subheader("⚙️ 检索模式")
    mode = st.radio(
        "选择检索模式",
        ["fast", "accurate"],
        index=0 if st.session_state.mode == "fast" else 1,
        format_func=lambda m: "快速（纯向量 ~19s）" if m == "fast" else "精确（向量+BM25+reranker）",
        help="快速：纯向量 top-5，秒级响应，适合大多数问题。"
        "精确：向量召回 top-50 + 自实现 BM25 词面召回，RRF 融合后再经 bge-reranker 精排，"
        "单题约 1~2 分钟，专治「ACID / 2NF / 死锁」等纯向量词面匹配弱的精确术语难 query。",
    )
    st.session_state.mode = mode
    if mode == "accurate":
        if not config.HYBRID_ENABLED and not config.RERANKER_ENABLED:
            st.warning("HYBRID_ENABLED 与 RERANKER_ENABLED 均为 false，精确模式退化为纯向量。")
        elif not config.RERANKER_ENABLED:
            st.info("精确模式当前为「向量 + BM25 融合」（未启用 reranker）；"
                    "设 RERANKER_ENABLED=true 可叠加 bge-reranker 精排。")

    st.markdown("---")

    # ---- 上传课程资料 ----
    st.subheader("上传课程资料")
    uploaded_files = st.file_uploader("PDF / TXT / MD / DOCX", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
    if uploaded_files:
        data_dir = Path(config.DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        saved_paths = []
        for file in uploaded_files:
            dest = data_dir / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(dest)
            st.success(f"已保存: {file.name}")
        if st.button("⤴️ 增量索引上传文件", type="primary"):
            # P1-4：基础库为空（首次/清空后）时退化为全量重建，避免 data/ 已有文件被静默漏索引；
            # 基础库已存在则只增量 upsert 刚上传的文件（稳定 id 自动去重，不重建全库）。
            if get_vector_store().count() == 0:
                with st.spinner("首次建库：从 data/ 全量重建索引..."):
                    chunks = load_and_chunk_documents()
                    get_vector_store(chunks)
                st.success("首次建库完成！")
            else:
                with st.spinner("正在增量索引（仅新上传文件）..."):
                    total = 0
                    for p in saved_paths:
                        total += add_documents(chunk_file(p))
                    st.success(f"增量索引完成！新增/覆盖 {total} 块")
            st.rerun()

    st.markdown("---")
    st.subheader("索引管理")
    st.caption("基础库已建好时，上方上传=增量追加；此按钮用于从 data/ 全量重建（忽略已上传但未索引的文件差异）。")
    if st.button("🔄 全量重建索引（data/ 全部）", type="primary", key="reindex_btn"):
        with st.spinner("正在从 data/ 重建索引..."):
            try:
                chunks = load_and_chunk_documents()
                get_vector_store(chunks)
                st.success(f"索引完成！共 {len(chunks)} 块")
                st.rerun()
            except Exception as e:
                st.error(f"索引失败: {e}")

    st.markdown("---")
    st.subheader("当前文档")
    data_dir = Path(config.DATA_DIR)
    current_files = [f.name for f in data_dir.glob("*.*")] if data_dir.exists() else []
    if current_files:
        for fname in current_files:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("🗑", key=f"del_{fname}", help="从索引中删除该文档（其余不受影响）"):
                try:
                    (data_dir / fname).unlink()
                    with st.spinner("正在从索引移除该文档..."):
                        # P1-4：只删该 source 的块，其余文档检索不受影响（不再全量重建）
                        delete_by_source(str(data_dir / fname))
                    st.success(f"已删除并移出索引：{fname}")
                    st.rerun()
                except Exception as e:
                    st.error(f"错误: {e}")
    else:
        st.info("还没有文档，先上传或把资料放进 data/ 目录。")

    st.markdown("---")
    st.subheader("系统提示词")
    st.caption("使用默认安全提示词" if st.session_state.custom_prompt is None else "已启用自定义提示词")
    custom_input = st.text_area("自定义提示词（留空用默认）", value="", height=120,
                                 placeholder="在此输入额外指令...\n{context} 与 {question} 为必填占位符")
    col_a, col_b = st.columns(2)
    if col_a.button("应用自定义"):
        st.session_state.custom_prompt = custom_input.strip() or None
        st.success("已应用" if custom_input.strip() else "已恢复默认")
        st.rerun()
    if col_b.button("恢复默认"):
        st.session_state.custom_prompt = None
        st.success("已恢复默认")
        st.rerun()

    st.markdown("---")
    # ---- P0-2 清空对话 ----
    st.subheader("💬 对话")
    if st.button("🧹 清空对话", key="clear_btn"):
        st.session_state.messages = []
        if HISTORY_FILE.exists():
            try:
                HISTORY_FILE.unlink()
            except Exception:
                pass
        st.rerun()

# ==================== 主区域 ====================
st.title("📚 CourseRAG")
st.markdown("基于本地 Ollama 的**专业课 / 课本问答**助手（数据不出本机）")
if st.session_state.mode == "accurate":
    rerank_part = " + bge-reranker 精排" if config.RERANKER_ENABLED else ""
    st.caption("检索策略：向量召回 top-%d + BM25 融合%s top-%d（精确模式）" % (
        config.RETRIEVE_K, rerank_part, config.RERANKER_TOP_N))
else:
    st.caption("检索策略：纯向量召回 top-%d（快速模式）" % config.RERANKER_TOP_N)

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        # P1-2 内联引用：助手回答里的 [n] 渲染为可点击锚点，定位到下方来源
        if role == "assistant" and message.get("citations"):
            st.markdown(
                render_answer_html(message["content"], len(message["citations"]), idx),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(message["content"])
        # 引用溯源：带排名 / 相关性分数 / 原文偏移，可展开核对
        if "citations" in message and message["citations"]:
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in message["citations"]:
                    score_txt = f" · 相关性 **{cit['score']:.3f}**" if cit.get("score") is not None else ""
                    # 锚点：与回答里的 [n] 对应，点击可跳转定位
                    st.markdown(
                        f'<div id="m{idx}-src-{cit["rank"]}"></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**#{cit['rank']} {cit['source']}**{score_txt}  `offset={cit.get('start_index')}`",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"```text\n{cit['content'][:500]}{'...' if len(cit['content']) > 500 else ''}\n```",
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("问任何专业课问题..."):
    # 先用「已有历史」拼上下文（不含本次提问），再追加用户消息
    history_text = build_history_text(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 本条助手消息落盘后的索引 = 当前长度（用户已追加），用于锚点 id 去重
    mid = len(st.session_state.messages)
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            accurate = st.session_state.mode == "accurate"

            # ---- P0-3 分阶段进度（覆盖 检索+重排+生成，结束显式标记完成）----
            # 统一走 retrieve_and_rerank（fast=纯向量 top-5；accurate=向量+BM25 融合+可选 reranker），
            # 把同一份 docs 既给引用展示、又传给链生成，杜绝重复检索与来源不一致。
            with st.status("正在处理…", expanded=True) as status:
                if accurate:
                    status.update(label="① 向量召回 + BM25 融合 top-%d …" % config.RETRIEVE_K)
                else:
                    status.update(label="① 向量召回 top-%d …" % config.RERANKER_TOP_N)
                docs = retrieve_and_rerank(prompt, force_no_rerank=not accurate)
                if accurate and config.RERANKER_ENABLED:
                    status.update(label="② bge-reranker 精排 top-%d …" % config.RERANKER_TOP_N)
                status.update(label="③ 生成回答…", state="running")

                # 引用溯源结构
                citations = []
                for rank, doc in enumerate(docs, 1):
                    source = Path(doc.metadata.get("source", "unknown")).name
                    content = doc.page_content.strip()
                    score = doc.metadata.get("rerank_score") or doc.metadata.get("hybrid_score")
                    start = doc.metadata.get("start_index")
                    citations.append({
                        "rank": rank,
                        "source": source,
                        "content": content,
                        "score": score,
                        "start_index": start,
                    })

                # 构建带历史记忆的链并生成（复用上方同一份 docs，引用与上下文一致）
                rag_result = build_rag_chain(
                    custom_prompt=st.session_state.custom_prompt,
                    history=history_text,
                )
                chain = rag_result["chain"]
                response = chain.invoke(prompt, docs=docs)
                full_response = response
                # 显式标记完成，避免阶段标签「③ 生成回答…」残留为 running
                status.update(label="✅ 回答完成", state="complete")

            # 流式回显（在状态框外，避免与阶段进度混淆）
            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[: i + 1] + "▋")
            # P1-2 内联引用：把 [n] 渲染为可点击锚点（仅合法编号），定位到下方来源
            message_placeholder.markdown(
                render_answer_html(full_response, len(citations), mid),
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations,
            })
            # ---- P0-2 落盘 ----
            save_history(st.session_state.messages)

            # ---- P1-3 查询 / 接地日志（每轮追加一行 .query_log.jsonl）----
            log_query(
                question=prompt,
                mode="精确" if accurate else "快速",
                citations=citations,
                answer=full_response,
            )

            # 展示引用
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in citations:
                    score_txt = f" · 相关性 **{cit['score']:.3f}**" if cit.get("score") is not None else ""
                    # 锚点：与回答里的 [n] 对应，点击可跳转定位
                    st.markdown(
                        f'<div id="m{mid}-src-{cit["rank"]}"></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**#{cit['rank']} {cit['source']}**{score_txt}  `offset={cit.get('start_index')}`",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"```text\n{cit['content'][:500]}{'...' if len(cit['content']) > 500 else ''}\n```",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"出错了: {str(e)}")
            st.info(
                "请确认 Ollama 已启动（ollama serve）且模型已拉取；"
                "若启用了 reranker，需先 pip install sentence-transformers 并联网下载模型权重一次。"
            )
