"""
app.py — Streamlit 演示界面（改编自 RAG-KnowBot + W2 引用溯源增强）。
保留引用溯源（Sources 折叠面板），并升级为：带排名 / 相关性分数 / 原文偏移。

P0 改造（2026-08-18）：
  - P0-1 启动自检：侧边栏顶部状态条（依赖 / Ollama / Reranker 权重），消灭神秘转圈。
  - P0-2 对话记忆：最近 N 轮对话喂给 LLM；历史落盘 .chat_history.json，刷新不丢；可清空。
  - P0-3 检索体验：分阶段进度（向量召回 → reranker 重排 → 生成）；「快速/精确」双模式
          （快速=纯向量 top-5 ~19s 默认；精确=reranker，救难 query）。

注意：删除文件后会用 data/ 下剩余文件整体重建索引（原项目行为，未改）。
"""
import streamlit as st
import os
import json

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
    retrieve_vector,
    rerank_documents,
)
import config
from health import collect_status

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

    # ---- P0-3 检索模式 ----
    st.subheader("⚙️ 检索模式")
    mode = st.radio(
        "选择检索模式",
        ["fast", "accurate"],
        index=0 if st.session_state.mode == "fast" else 1,
        format_func=lambda m: "快速（纯向量 ~19s）" if m == "fast" else "精确（reranker 救难 query）",
        help="快速：纯向量 top-5，秒级响应，适合大多数问题。"
        "精确：向量召回 top-50 + bge-reranker 精排，单题约 1~2 分钟，"
        "专治「区别类」等向量召回失败的难 query。",
    )
    st.session_state.mode = mode
    if mode == "accurate" and not config.RERANKER_ENABLED:
        st.warning("config.RERANKER_ENABLED=false，精确模式将退化为纯向量。")

    st.markdown("---")

    # ---- 上传课程资料 ----
    st.subheader("上传课程资料")
    uploaded_files = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploaded_files:
        data_dir = Path(config.DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        for file in uploaded_files:
            with open(data_dir / file.name, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"已保存: {file.name}")
        if st.button("🔄 索引 / 更新", type="primary"):
            with st.spinner("正在处理文档..."):
                chunks = load_and_chunk_documents()
                get_vector_store(chunks)
                st.success("索引完成！")
                st.rerun()

    st.subheader("当前文档")
    data_dir = Path(config.DATA_DIR)
    current_files = [f.name for f in data_dir.glob("*.*")] if data_dir.exists() else []
    if current_files:
        for fname in current_files:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("🗑", key=f"del_{fname}", help="删除文件并重建索引"):
                try:
                    (data_dir / fname).unlink()
                    st.success(f"已删除 {fname}")
                    with st.spinner("重建剩余文档索引..."):
                        chunks = load_and_chunk_documents()
                        get_vector_store(chunks)
                    st.rerun()
                except Exception as e:
                    st.error(f"错误: {e}")
    else:
        st.info("还没有文档，先上传或把资料放进 data/ 目录。")

    st.markdown("---")
    st.subheader("索引管理")
    st.caption("往 data/ 手动增删文件后，点这里重建向量库（不必走上方上传）")
    if st.button("🔄 更新 / 重建索引", type="primary", key="reindex_btn"):
        with st.spinner("正在从 data/ 重建索引..."):
            try:
                chunks = load_and_chunk_documents()
                get_vector_store(chunks)
                st.success(f"索引完成！共 {len(chunks)} 块")
                st.rerun()
            except Exception as e:
                st.error(f"索引失败: {e}")

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
    st.caption("检索策略：向量召回 top-%d + bge-reranker 精排 top-%d（精确模式）" % (
        config.RETRIEVE_K, config.RERANKER_TOP_N))
else:
    st.caption("检索策略：纯向量召回 top-%d（快速模式）" % config.RERANKER_TOP_N)

for message in st.session_state.messages:
    role = message["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        # 引用溯源：带排名 / 相关性分数 / 原文偏移，可展开核对
        if "citations" in message and message["citations"]:
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in message["citations"]:
                    score_txt = f" · 相关性 **{cit['score']:.3f}**" if cit.get("score") is not None else ""
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
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            accurate = st.session_state.mode == "accurate"
            k = config.RETRIEVE_K if accurate else config.RERANKER_TOP_N

            # ---- P0-3 分阶段进度（覆盖 检索+重排+生成，结束显式标记完成）----
            with st.status("正在处理…", expanded=True) as status:
                status.update(label="① 向量召回 top-%d …" % k)
                docs = retrieve_vector(prompt, k=k)
                if accurate and config.RERANKER_ENABLED:
                    status.update(label="② bge-reranker 精排 top-%d …" % config.RERANKER_TOP_N)
                    docs = rerank_documents(prompt, docs, top_n=config.RERANKER_TOP_N)
                else:
                    docs = docs[: config.RERANKER_TOP_N]
                status.update(label="③ 生成回答…", state="running")

                # 引用溯源结构
                citations = []
                for rank, doc in enumerate(docs, 1):
                    source = Path(doc.metadata.get("source", "unknown")).name
                    content = doc.page_content.strip()
                    score = doc.metadata.get("rerank_score")
                    start = doc.metadata.get("start_index")
                    citations.append({
                        "rank": rank,
                        "source": source,
                        "content": content,
                        "score": score,
                        "start_index": start,
                    })

                # 构建带历史记忆的链并生成
                rag_result = build_rag_chain(
                    custom_prompt=st.session_state.custom_prompt,
                    history=history_text,
                )
                chain = rag_result["chain"]
                response = chain.invoke(prompt)
                full_response = response
                # 显式标记完成，避免阶段标签「③ 生成回答…」残留为 running
                status.update(label="✅ 回答完成", state="complete")

            # 流式回显（在状态框外，避免与阶段进度混淆）
            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[: i + 1] + "▋")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations,
            })
            # ---- P0-2 落盘 ----
            save_history(st.session_state.messages)

            # 展示引用
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in citations:
                    score_txt = f" · 相关性 **{cit['score']:.3f}**" if cit.get("score") is not None else ""
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
