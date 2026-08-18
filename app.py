"""
app.py — Streamlit 演示界面（改编自 RAG-KnowBot + W2 引用溯源增强）。
保留引用溯源（Sources 折叠面板），并升级为：带排名 / 相关性分数 / 原文偏移。
注意：删除文件后会用 data/ 下剩余文件整体重建索引（原项目行为，未改）。
"""
import streamlit as st
import os

# reranker 权重（bge-reranker-v2-m3）加载需访问 HuggingFace。本机 huggingface.co 不可达会
# 超时卡死（页面一直转）。必须在 import rag_core / langchain（会间接 import huggingface_hub）
# 之前设置——否则 huggingface_hub 在导入时就把 HF_ENDPOINT 捕获为默认 huggingface.co。
# 默认走镜像；若本机已缓存权重则强制离线，彻底不碰网络（权重已下载在 ~/.cache/huggingface）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_rerank_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
)
if os.path.isdir(_rerank_cache):
    os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from rag_core import load_and_chunk_documents, get_vector_store, build_rag_chain
import config

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="CourseRAG - 专业课问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 会话状态 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📚 CourseRAG 控制台")

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

# ==================== 主区域 ====================
st.title("📚 CourseRAG")
st.markdown("基于本地 Ollama 的**专业课 / 课本问答**助手（数据不出本机）")
if config.RERANKER_ENABLED:
    st.caption("检索策略：向量召回 top-%d + bge-reranker 精排 top-%d" % (config.RETRIEVE_K, config.RERANKER_TOP_N))
else:
    st.caption("检索策略：纯向量召回 top-%d（reranker 已关闭）" % config.RETRIEVE_K)

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            rag_result = build_rag_chain(custom_prompt=st.session_state.custom_prompt)
            chain = rag_result["chain"]
            retrieve_and_rerank = rag_result["retrieve_and_rerank"]

            # 先检索 + 重排拿到来源文档（已按相关性排序，带 rerank 分数）
            docs = retrieve_and_rerank(prompt)
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
                    "score": score,         # reranker 相关性分
                    "start_index": start,   # 该块在原文的字符偏移
                })

            response = chain.invoke(prompt)
            full_response = response

            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[: i + 1] + "▋")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations,
            })
        except Exception as e:
            st.error(f"出错了: {str(e)}")
            st.info(
                "请确认 Ollama 已启动（ollama serve）且模型已拉取；"
                "若启用了 reranker，需先 pip install sentence-transformers 并联网下载模型权重一次。"
            )
