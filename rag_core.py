"""
rag_core.py — RAG 核心（W1 三件套 + W2 重排增强 + 引用溯源封装）

2026-08-18 重构：彻底移除 langchain 全家桶，改为直接依赖：
  - 嵌入 / 生成：ollama 官方 SDK（ollama.Client）
  - 向量库：chromadb 原生客户端（PersistentClient）
  - 切分：自实现 RecursiveCharacterTextSplitter（含中文标点分隔符 + start_index）
  - 文档加载：pypdf / python-docx / 原生读文件（不再经 langchain_community loaders）
  - 链：手工拼 prompt + ollama.chat，不再用 LCEL

为什么要去 langchain（见交接文档）：
  langchain 只是薄封装，却带来『全家桶版本必须一致』的脆弱性
  （is_data_content_block 等符号冲突），并间接 import huggingface_hub 制造
  HF_ENDPOINT 时序坑。去掉后依赖从十几个包瘦成 ollama + chromadb + sentence-transformers，
  可复现性直接上一个台阶，也更贴『自己实现关键组件』的叙事。

对外接口保持稳定（app.py / eval_local.py 无需改动）：
  load_and_chunk_documents / get_vector_store / retrieve_vector /
  retrieve_and_rerank / build_rag_chain（返回 {'chain','retrieve_and_rerank'}）/
  ollama_chat（供 eval_local 当裁判复用）
"""
import os
import hashlib
from pathlib import Path

import ollama
import chromadb

import config
from reranker import rerank_documents


# ====================== 0. 轻量客户端 / Document ======================
# ollama 0.6.2 的模块级函数不接收 base_url，统一用 Client 实例
_client = ollama.Client(host=config.OLLAMA_BASE_URL)


class Document:
    """最小文档对象：只保留 page_content + metadata，替代 langchain_core.documents.Document。
    reranker / app / eval_local 都只读写这两个属性，故完全兼容。"""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def ollama_chat(messages, temperature=None):
    """统一封装 ollama 对话调用，返回模型输出的文本字符串。"""
    opts = {"temperature": temperature} if temperature is not None else None
    resp = _client.chat(model=config.LLM_MODEL, messages=messages, options=opts)
    return resp["message"]["content"]


# ====================== 1. 加载 + 切分 ======================
_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""]


def _load_text(path: Path):
    """按后缀加载单文件为纯文本；不支持或读取失败返回 None（跳过不中断建库）。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        if suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[warn] 跳过无法读取的 {path}: {e}")
    return None


def _split_text(text, chunk_size, chunk_overlap, separators):
    """递归切分（模仿 RecursiveCharacterTextSplitter），返回文本块列表。"""
    if text == "":
        return []

    # 选最粗的可行分隔符（优先段落/换行，再中文标点，最终回退到逐字）
    separator = separators[-1]
    for s in separators:
        if s == "":
            separator = s
            break
        if s in text:
            separator = s
            break

    if separator == "":
        splits = list(text)
    else:
        raw = text.split(separator)
        # 把分隔符重新贴回除最后一段外的每段，保持原文连续性
        splits = [s + separator for s in raw[:-1]] + [raw[-1]]

    good_splits = []
    for s in splits:
        if s == "":
            continue
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if len(separators) > 1:
                good_splits.extend(_split_text(s, chunk_size, chunk_overlap, separators[1:]))
            else:
                good_splits.append(s)

    return _merge_splits(good_splits, chunk_size, chunk_overlap)


def _merge_splits(splits, chunk_size, chunk_overlap):
    """按 chunk_size 合并，并在块间保留 chunk_overlap 字符重叠（避免切断句子）。"""
    docs = []
    current = ""
    for piece in splits:
        if current and len(current) + len(piece) > chunk_size:
            docs.append(current)
            # 保留末尾 chunk_overlap 字符作为下一段开头
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:]
            else:
                current = ""
        current += piece
    if current:
        docs.append(current)
    return docs


def _split_with_index(text, chunk_size, chunk_overlap, separators):
    """切分并记录每块在原文中的偏移（start_index），供引用溯源定位。"""
    chunks = _split_text(text, chunk_size, chunk_overlap, separators)
    result = []
    cursor = 0
    for ch in chunks:
        start = text.find(ch, cursor)
        if start == -1:
            start = cursor
        result.append((ch, start))
        cursor = start + len(ch)
    return result


def load_and_chunk_documents(directory=config.DATA_DIR):
    # 支持 PDF / DOCX / TXT / MD；TXT/MD 强制 utf-8（Windows 中文坑，直接写死避免再踩）
    directory = Path(directory)
    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".pdf", ".txt", ".md", ".docx"):
            continue
        text = _load_text(path)
        if not text or not text.strip():
            continue
        for chunk, start in _split_with_index(
            text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
        ):
            documents.append(
                Document(page_content=chunk, metadata={"source": str(path), "start_index": start})
            )
    print(f"Loaded and chunked {len(documents)} pieces from documents.")
    return documents


# ====================== 2. 嵌入 + 向量库（带缓存） ======================
class _OllamaEmbeddingFunction:
    """chromadb 1.5.x 兼容的嵌入函数：封装 ollama 的 nomic-embed-text。
    新式接口要求 __call__(self, input) 接受文本列表并返回 embedding 列表。"""

    def __init__(self, client, model):
        self._client = client
        self._model = model

    def name(self):
        # chromadb 1.5.x 要求嵌入函数提供 name()，用于持久化配置冲突校验
        return "ollama"

    def __call__(self, input):
        # 建库时 chromadb 用此嵌入文档（input 为文本列表）
        resp = self._client.embed(model=self._model, input=input)
        return resp["embeddings"]

    def embed_query(self, input):
        # 查询时 chromadb 1.5.x 单独调 embed_query（input 为查询文本列表，非单字符串）
        # 直接复用 __call__（批量 client.embed 对列表/字符串都适用）
        return self(input)


def get_embeddings():
    # 统一实例化嵌入函数，建库与查询共用（维度一致）
    return _OllamaEmbeddingFunction(_client, config.EMBEDDING_MODEL)


_COLLECTION_NAME = "course_rag"  # 中性集合名（不再沿用 langchain_chroma 遗留的 'langchain' 名，避免旧 ef 元数据冲突）
_chroma_client = None
_vector_store = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
    return _chroma_client


def get_vector_store(chunks=None):
    """chunks 不为 None → 重建并落盘；为 None → 懒加载已有库（问答阶段用）。"""
    global _vector_store
    ef = get_embeddings()
    client = _get_chroma_client()

    if chunks is not None:
        # 全量重建：先删旧 collection 再建，避免删文件后旧文档残留（原 langchain 版有此隐患）
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        col = client.get_or_create_collection(_COLLECTION_NAME, embedding_function=ef)
        ids = [
            hashlib.md5(
                f"{d.metadata['source']}::{d.metadata['start_index']}".encode("utf-8")
            ).hexdigest()
            for d in chunks
        ]
        col.upsert(
            ids=ids,
            documents=[d.page_content for d in chunks],
            metadatas=[d.metadata for d in chunks],
        )
        _vector_store = col
        print("Vector store created and saved.")
        return _vector_store

    if _vector_store is None:
        # 加载已有库；若旧 chroma 格式不兼容导致失败，则清空目录后重建（本地缓存可一键重建）
        try:
            _vector_store = client.get_or_create_collection(
                _COLLECTION_NAME, embedding_function=ef
            )
            # 触发一次轻量访问，确认集合可正常打开（旧格式异常会在 query 前暴露）
            _vector_store.count()
        except Exception as e:
            print(f"[warn] 已有向量库无法加载（{e}），将清空并重建：{config.PERSIST_DIRECTORY}")
            import shutil
            shutil.rmtree(config.PERSIST_DIRECTORY, ignore_errors=True)
            _chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
            _vector_store = _chroma_client.get_or_create_collection(
                _COLLECTION_NAME, embedding_function=ef
            )
        # 集合为空但本地有资料 → 自动重建（切换集合名 / 首次部署后免手动点「重建索引」）
        if _vector_store.count() == 0:
            chunks = load_and_chunk_documents()
            if chunks:
                return get_vector_store(chunks)
        print("Loaded existing vector store.")
    return _vector_store


# ====================== 3. 检索 + 重排（两段式） ======================
def retrieve_vector(question, k=None):
    """
    仅做向量召回（双塔粗排），返回 top-k 个 Document（已按相似度降序）。
    单独暴露出来，供 app.py 分阶段显示进度（先召回、再重排），也供「快速模式」直出。
    """
    k = k or config.RETRIEVE_K
    collection = get_vector_store()
    res = collection.query(query_texts=[question], n_results=k)
    texts = res["documents"][0]
    metas = res["metadatas"][0]
    return [
        Document(page_content=t, metadata=m if m is not None else {})
        for t, m in zip(texts, metas)
    ]


def retrieve_and_rerank(question, force_no_rerank=False):
    """
    召回 RETRIEVE_K 条候选 → （可选）bge-reranker 精排 → 取 top N。
    返回按相关性排序的 Document 列表，每个 doc.metadata 含 rerank_score（若启用）。
    UI 用它做引用展示；RAG 链也用它拼 context——保证『看到的来源』就是『喂给模型的来源』。
    force_no_rerank=True 时无视配置强制走纯向量（「快速模式」用）。
    """
    docs = retrieve_vector(question)

    if config.RERANKER_ENABLED and not force_no_rerank:
        docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
    else:
        # 基线对照：召回候选已按向量相似度从高到低排序，直接截取前 TOP_N 条，
        # 保证 OFF 与 ON 喂给 LLM 的文档数一致（都是 RERANKER_TOP_N），
        # 让 A/B 差值纯粹反映 reranker 的贡献，不被"上下文条数不同"污染。
        docs = docs[: config.RERANKER_TOP_N]
    return docs


# ====================== 4. 拼 RAG 链（含引用） ======================
SYSTEM_RULES = """你是一位严谨、准确的高校课程助教，专门解答专业课与课本问题。
请严格遵守以下规则：
1. 只能依据下面提供的【上下文】作答，严禁使用上下文之外的知识，也不得凭记忆编造。
2. 合理使用 Markdown（加粗、列表等）让回答清晰易读。
3. 若【上下文】信息不足、前后矛盾，或包含玩笑/错误内容，请直接说明"根据提供的资料无法回答"，不要猜测。
4. 回答语言必须与用户提问保持一致（用户用中文提问，就用中文回答）。
5. 回答要紧扣问题，避免堆砌与问题无关的内容。"""


def _build_prompt(rules, history, context, question):
    """拼装最终 prompt（与去 langchain 前 ChatPromptTemplate 的模板逐字一致）。"""
    return f"""{rules}

（以下是之前的对话，仅作上下文参考；回答仍须严格以【上下文】为准）
{history}

上下文：
{context}

问题：
{question}

回答："""


class RagChain:
    """替代 LCEL 链：持有 history / custom_prompt，invoke(question) 走 向量召回→重排→生成。"""

    def __init__(self, custom_prompt=None, history=""):
        self.custom_prompt = custom_prompt
        self.history = history

    def invoke(self, question):
        docs = retrieve_and_rerank(question)
        context = "\n\n".join(d.page_content for d in docs)

        if self.custom_prompt and self.custom_prompt.strip():
            rules = f"""{SYSTEM_RULES}

【用户补充要求】
{self.custom_prompt.strip()}"""
        else:
            rules = SYSTEM_RULES

        prompt = _build_prompt(rules, self.history, context, question)
        return ollama_chat(
            [{"role": "user", "content": prompt}], temperature=config.LLM_TEMPERATURE
        )


def build_rag_chain(custom_prompt=None, history=""):
    """
    返回 {'chain': RagChain 实例, 'retrieve_and_rerank': 检索+重排函数}
    retriever 不再单独暴露，统一走 retrieve_and_rerank，保证 UI 与链来源一致。
    history: 最近若干轮对话文本（由 app.py 拼好传入），让助手有上下文记忆。
    """
    return {"chain": RagChain(custom_prompt=custom_prompt, history=history),
            "retrieve_and_rerank": retrieve_and_rerank}
