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
  retrieve_and_rerank（P1-1 起内部支持 hybrid 第一段）/ hybrid_retrieve（向量+BM25+RRF）/
  build_rag_chain（返回 {'chain','retrieve_and_rerank'}）/ ollama_chat（供 eval_local 当裁判复用）
  RagChain.invoke(question, docs=None)：docs 可不传（自行检索）或由调用方传入以复用同一份来源
  P1-4 增量索引：chunk_file(path) / add_documents(chunks)（稳定 id 增量 upsert 去重）/
  delete_by_source(source)（按 source 删单文档）。get_vector_store(chunks) 仍保留为全量重建入口。
"""
import os
import math
import re
import hashlib
from pathlib import Path

# 纯检索算法（零依赖）抽到 rag_search_utils，便于独立单测（见 test_core.py）；
# 这里重新导出，对外接口 / 内部调用（hybrid_retrieve、add_documents）均不变。
from rag_search_utils import (
    _tokenize,
    _BM25,
    _bm25_rank,
    _rrf_fuse,
    _stable_ids,
)

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
        _invalidate_all_chunks()  # 全量重建后 BM25 通道缓存强制刷新
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


# ====================== 2.1 增量索引 + 去重（P1-4） ======================
def chunk_file(path):
    """切分单个文件为 Document 列表（P1-4 增量索引用；metadata 同 load_and_chunk_documents）。"""
    path = Path(path)
    text = _load_text(path)
    if not text or not text.strip():
        return []
    docs = []
    for chunk, start in _split_with_index(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS):
        docs.append(
            Document(page_content=chunk, metadata={"source": str(path), "start_index": start})
        )
    return docs


def add_documents(new_chunks):
    """
    P1-4 增量 upsert：只插入/覆盖给定块，不删除已有库。
    稳定 id = md5(source::start_index)，因此重复上传同一文件 → 同 id 覆盖，不产生重复块。
    返回实际 upsert 的块数。
    """
    if not new_chunks:
        return 0
    ef = get_embeddings()
    client = _get_chroma_client()
    col = client.get_or_create_collection(_COLLECTION_NAME, embedding_function=ef)
    ids = _stable_ids(new_chunks)
    col.upsert(
        ids=ids,
        documents=[d.page_content for d in new_chunks],
        metadatas=[d.metadata for d in new_chunks],
    )
    global _vector_store
    _vector_store = col
    print(f"Incremental upsert: {len(new_chunks)} chunks (dedup by stable id).")
    _invalidate_all_chunks()
    return len(new_chunks)


def delete_by_source(source):
    """
    P1-4 按源文件删除（删除该 source 的全部块）。
    source 用绝对路径字符串，须与切分时 metadata['source'] 完全一致。
    只移除该文档，其余文档不受影响（对比旧的『删文件即全量重建』）。
    """
    col = get_vector_store()  # 确保集合已加载
    col.delete(where={"source": str(source)})
    _invalidate_all_chunks()
    print(f"Deleted chunks where source={source}")


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


# ====================== 3.1 混合检索（向量 + BM25 双通道，RRF 融合） ======================
_all_chunks_cache = None
_all_chunks_count = -1


def _invalidate_all_chunks():
    """增删文档后调用：强制下次查询重新拉取全库（BM25 通道同步最新语料）。"""
    global _all_chunks_cache, _all_chunks_count
    _all_chunks_cache = None
    _all_chunks_count = -1


def _get_all_chunks():
    """从向量库拉取全部块（含 metadata），供 BM25 全库独立召回使用。

    带缓存：collection.count() 变化时自动刷新（增删/重建索引后 count 必变），
    避免每次查询都全库拉取。返回 List[Document]（page_content + metadata）。
    """
    global _all_chunks_cache, _all_chunks_count
    col = get_vector_store()
    cnt = col.count()
    if _all_chunks_cache is None or cnt != _all_chunks_count:
        data = col.get(include=["documents", "metadatas"])
        _all_chunks_cache = [
            Document(page_content=t, metadata=m if m is not None else {})
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        _all_chunks_count = cnt
    return _all_chunks_cache


def hybrid_retrieve(question, k=config.RETRIEVE_K):
    """真混合检索：两个独立召回通道，各自全库召回 top-K，RRF 融合取 top-N。

    - 通道 A：向量召回（双塔粗排，语义匹配）—— Chroma 全库
    - 通道 B：BM25 词面召回（精确术语匹配，如 ACID / 2NF / 死锁）—— 对全库打分
    两路结果用倒数排名融合（RRF）互补，返回 top-N（RERANKER_TOP_N）。

    相比 v1「向量 top-50 池内 BM25 重排」的根本区别：
    旧版 BM25 只对已召回的 50 块重排，正确块若不在向量 top-50 就永远捞不到；
    新版 BM25 对全库打分，正确块即使向量排名很低，只要词面匹配也能独立召回进池。
    """
    vector_docs = retrieve_vector(question, k=k)  # 通道 A：向量全库 top-K
    if not vector_docs:
        return []
    all_chunks = _get_all_chunks()
    bm25_docs = _bm25_rank(question, all_chunks, top_n=config.BM25_TOP_K)  # 通道 B：BM25 全库 top-K
    return _rrf_fuse([vector_docs, bm25_docs], top_n=config.RERANKER_TOP_N)


def retrieve_and_rerank(question, force_no_rerank=False):
    """
    统一检索入口（UI 与 RAG 链共用，保证『看到的来源』=『喂给模型的来源』）：
      - force_no_rerank=True：纯向量 top-N（「快速模式」/ 评估纯向量对照），不进 hybrid 也不进 reranker。
      - 否则：
          HYBRID_ENABLED=True  → 先做向量+BM25 的 RRF 融合得到 top-N 候选；
                                若 RERANKER_ENABLED=True 再对候选做 bge-reranker 精排。
          HYBRID_ENABLED=False → 原两段式（向量 top-K → reranker top-N）；reranker 关则纯向量 top-N。
    返回按相关性排序的 Document 列表；reranker 开启时 metadata 含 rerank_score，否则含 hybrid_score。
    """
    if force_no_rerank:
        # 纯向量对照：直接取 top-N，保证 A/B 差值纯粹反映 reranker/hybrid 的贡献。
        return retrieve_vector(question, k=config.RERANKER_TOP_N)

    if config.HYBRID_ENABLED:
        docs = hybrid_retrieve(question)  # 已融合出 top-N
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
    else:
        docs = retrieve_vector(question, k=config.RETRIEVE_K)
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
        else:
            docs = docs[: config.RERANKER_TOP_N]
    return docs


# ====================== 4. 拼 RAG 链（含引用） ======================
SYSTEM_RULES = """你是一位严谨、准确的高校课程助教，专门解答专业课与课本问题。
请严格遵守以下规则：
1. 只能依据下面提供的【上下文】作答，严禁使用上下文之外的知识，也不得凭记忆编造。
   重要：若【上下文】缺少回答所需的某些信息，请直接说明"根据提供的资料无法回答/无法完整回答（缺 XX）"，
   绝不要用你自己的通用知识去补全答案——宁可少答，不可错答。
2. 合理使用 Markdown（加粗、列表等）让回答清晰易读。
3. 仅当【上下文】确实未提供回答所需信息、或上下文内部相互矛盾时，才说明"根据提供的资料无法回答"；不要猜测。注意：即使上下文描述的是示例、虚构或假设性概念，只要其中已给出明确答案，就必须基于上下文如实作答，不得以此为由拒答。
4. 回答语言必须与用户提问保持一致（用户用中文提问，就用中文回答）。
5. 回答要紧扣问题，避免堆砌与问题无关的内容。
6. 回答中每提及一条来自【上下文】的具体事实，必须紧跟来源标注 [n]（半角方括号，n 为下方『已知来源』中的编号，即上下文第 n 段）；同一句可标多个如 [1][2]。仅当某事实确实无法对应任何来源时才不标。严禁编造超出范围的序号（n 必须 ≤ 已知来源条数）。"""


def _build_prompt(rules, history, context, question, sources=None):
    """拼装最终 prompt（与去 langchain 前 ChatPromptTemplate 的模板逐字一致）。
    sources: 可选，已编号的『已知来源』列表（P1-2 内联引用用），与 context 段落顺序一一对应。
    """
    sources_block = ""
    if sources:
        sources_block = (
            "\n\n【已知来源】（共 %d 条，回答中引用事实请用 [n] 标注对应序号；"
            "n 即下方编号，也是上方上下文第 n 段）\n%s"
            % (len(sources), "\n".join(sources))
        )
    return f"""{rules}

（以下是之前的对话，仅作上下文参考；回答仍须严格以【上下文】为准）
{history}

上下文：
{context}{sources_block}

问题：
{question}

回答："""


class RagChain:
    """替代 LCEL 链：持有 history / custom_prompt，invoke(question) 走 向量召回→重排→生成。"""

    def __init__(self, custom_prompt=None, history=""):
        self.custom_prompt = custom_prompt
        self.history = history

    def invoke(self, question, docs=None):
        # docs 由调用方预计算时直接复用（UI 用同一份做引用，避免重复检索/来源不一致）；
        # 不传则自行检索（eval_local 等场景），保持向后兼容。
        if docs is None:
            docs = retrieve_and_rerank(question)
        context = "\n\n".join(d.page_content for d in docs)

        # P1-2 内联引用：把检索到的块按 1..N 编号，供 prompt 要求 LLM 用 [n] 标注来源。
        # 编号顺序与 context 段落顺序一致，app.py 引用区与之一一对应。
        sources = []
        for i, d in enumerate(docs, 1):
            name = Path(d.metadata.get("source", "unknown")).name
            off = d.metadata.get("start_index")
            sources.append(f"{i}. {name}（offset={off}）")

        if self.custom_prompt and self.custom_prompt.strip():
            rules = f"""{SYSTEM_RULES}

【用户补充要求】
{self.custom_prompt.strip()}"""
        else:
            rules = SYSTEM_RULES

        prompt = _build_prompt(rules, self.history, context, question, sources=sources)
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
