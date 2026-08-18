"""
rag_core.py — RAG 核心（W1 三件套 + W2 重排增强 + 引用溯源封装）

相对 W1 的改造：
  - retriever 召回 RETRIEVE_K 条候选（更多），再经 bge-reranker 精排到 RERANKER_TOP_N
  - 新增 retrieve_and_rerank()：返回『重排后、按相关性排序、带分数』的文档，UI 与链共用
  - 向量库加载做了模块级缓存，避免每次提问重复加载 Chroma
  - 引用溯源结构化为 Citation（source/content/score/start_index），直接可喂 RAGAS 评估
"""
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

import config
from reranker import rerank_documents


# ====================== 1. 加载 + 切分 ======================
def load_and_chunk_documents(directory=config.DATA_DIR):
    # 加载器分别吃 PDF / DOCX / TXT / MD；TXT/MD 强制 utf-8（你踩过的 Windows 中文坑，这里直接写死避免再踩）
    pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    md_loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    documents = pdf_loader.load() + txt_loader.load() + md_loader.load()

    # .docx 支持（python-docx 提取文本层）。旧版 .doc（Word 97-2003）python-docx 读不了，
    # 需先在 Word / LibreOffice 另存为 .docx 再放进来；这里只吃 .docx，读不了的告警跳过不中断建库。
    from pathlib import Path as _Path
    for docx_path in _Path(directory).rglob("*.docx"):
        try:
            import docx as _docx
            _d = _docx.Document(str(docx_path))
            _text = "\n".join(p.text for p in _d.paragraphs if p.text.strip())
            if _text.strip():
                documents.append(Document(page_content=_text, metadata={"source": str(docx_path)}))
        except Exception as e:
            print(f"[warn] 跳过无法读取的 docx {docx_path}: {e}")

    # RecursiveCharacterTextSplitter：按段落/句子递归切，比硬切更保语义
    # add_start_index=True：记录每块在原文中的偏移，引用溯源时能定位
    # separators 显式加入中文标点（。；！？，、），让切块优先落在中文句子/子句边界，
    # 而非在字中间硬切——这是英文默认分隔符对中文不友好时的关键修正。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Loaded and chunked {len(chunks)} pieces from documents.")
    return chunks


# ====================== 2. 嵌入 + 向量库（带缓存） ======================
def get_embeddings():
    # 统一在此实例化嵌入模型，retriever 与建库共用同一实例（维度必须一致）
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)


_vector_store = None


def get_vector_store(chunks=None):
    global _vector_store
    embeddings = get_embeddings()

    if chunks is not None:  # 有文档 → 新建并落盘，同时刷新缓存
        _vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.PERSIST_DIRECTORY,
        )
        print("Vector store created and saved.")
        return _vector_store

    if _vector_store is None:  # 无文档 → 懒加载已有库（问答阶段用）
        _vector_store = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        print("Loaded existing vector store.")
    return _vector_store


# ====================== 3. 检索 + 重排（两段式） ======================
def retrieve_and_rerank(question):
    """
    召回 RETRIEVE_K 条候选 → （可选）bge-reranker 精排 → 取 top N。
    返回按相关性排序的 Document 列表，每个 doc.metadata 含 rerank_score（若启用）。
    UI 用它做引用展示；RAG 链也用它拼 context——保证『看到的来源』就是『喂给模型的来源』。
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    docs = retriever.invoke(question)

    if config.RERANKER_ENABLED:
        docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
    else:
        # 基线对照：召回候选已按向量相似度从高到低排序，直接截取前 TOP_N 条，
        # 保证 OFF 与 ON 喂给 LLM 的文档数一致（都是 RERANKER_TOP_N），
        # 让 A/B 差值纯粹反映 reranker 的贡献，不被"上下文条数不同"污染。
        docs = docs[: config.RERANKER_TOP_N]
    return docs


# ====================== 4. 拼 RAG 链（含引用） ======================
def build_rag_chain(custom_prompt=None):
    """
    返回 {'chain': 可 invoke 的链, 'retrieve_and_rerank': 检索+重排函数}
    retriever 不再单独暴露，统一走 retrieve_and_rerank，保证 UI 与链来源一致。
    """
    llm = ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        base_url=config.OLLAMA_BASE_URL,
    )

    # 中文安全 prompt：强制『只基于上下文』+『不确定就说不确定』，这是防幻觉的关键。
    # 同时要求语言对齐（中文问→中文答），并避免堆砌无关内容。
    SYSTEM_RULES = """你是一位严谨、准确的高校课程助教，专门解答专业课与课本问题。
请严格遵守以下规则：
1. 只能依据下面提供的【上下文】作答，严禁使用上下文之外的知识，也不得凭记忆编造。
2. 合理使用 Markdown（加粗、列表等）让回答清晰易读。
3. 若【上下文】信息不足、前后矛盾，或包含玩笑/错误内容，请直接说明"根据提供的资料无法回答"，不要猜测。
4. 回答语言必须与用户提问保持一致（用户用中文提问，就用中文回答）。
5. 回答要紧扣问题，避免堆砌与问题无关的内容。"""

    # 用户补充要求：追加在规则之后，再统一接 Context/Question/Answer 三段，避免重复拼接
    if custom_prompt and custom_prompt.strip():
        rules = f"""{SYSTEM_RULES}

【用户补充要求】
{custom_prompt.strip()}"""
    else:
        rules = SYSTEM_RULES

    template = f"""{rules}

上下文：
{{context}}

问题：
{{question}}

回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        # 把检索到的若干块拼成一段 context 喂给 LLM
        return "\n\n".join(doc.page_content for doc in docs)

    # RunnableLambda 把『检索+重排』包成一个可组合节点，喂给 prompt 的 context 字段
    retrieval = RunnableLambda(lambda q: retrieve_and_rerank(q))

    chain = (
        {"context": retrieval | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retrieve_and_rerank": retrieve_and_rerank}
