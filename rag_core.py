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
import grounding  # P1 接地自检（grounding 仅依赖 config/ollama，无循环依赖）
import semantic_cache  # 统一问答入口的缓存查询/写入（仅依赖 config/ollama，无循环依赖）


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


def _extract_pdf_text(path: Path):
    """PDF 文字层提取：优先 PyMuPDF，回退 pypdf。

    G2（2026-08-28）：实测《微信小程序开发实战》303 页 pypdf 提取中文全为 CID
    乱码（正文不可读、向量通道失效），而 PyMuPDF 对同一 PDF 296/303 页正常提取。
    根因是 CID→Unicode 映射（ToUnicode CMap），非扫描版——换提取器即可免费修复，
    无需 OCR。PyMuPDF 缺失/提取过低时回退 pypdf（保持原扫描版检测行为）。
    返回 (text, n_pages)；text 为空字符串表示无文字层。
    """
    try:
        import pymupdf  # 1.28+ 官方命名
    except ImportError:
        try:
            import fitz as pymupdf  # 旧命名
        except ImportError:
            pymupdf = None
    if pymupdf is not None:
        try:
            doc = pymupdf.open(str(path))
            pages = [p.get_text() or "" for p in doc]
            avg = sum(len(t.strip()) for t in pages) / max(len(pages), 1)
            if avg >= config.PDF_MIN_CHARS_PER_PAGE:
                return "\n".join(pages), len(pages)
            # PyMuPDF 也提不出 → 真扫描/伪文字层，落到 pypdf 再确认一次
        except Exception:
            pass
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages), len(pages)


def _load_text(path: Path):
    """按后缀加载单文件为纯文本；不支持或读取失败返回 None（跳过不中断建库）。

    T6（2026-08-22）：PDF 加载后做文字层检测——扫描版 PDF（无文字层）此前会
    静默返回空内容、对检索零贡献且无提示（data/教材 实测 7/7 全中）。现在均字符
    低于 config.PDF_MIN_CHARS_PER_PAGE 时告警并跳过；另新增 .pptx 课件支持。
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            text, n_pages = _extract_pdf_text(path)
            avg = (len(text.strip()) / n_pages) if n_pages else 0
            if avg < config.PDF_MIN_CHARS_PER_PAGE:
                print(
                    f"[warn] 跳过疑似扫描版/无文字层 PDF: {path.name} "
                    f"(均 {avg:.0f} 字符/页，阈值 {config.PDF_MIN_CHARS_PER_PAGE})"
                )
                return None
            return text
        if suffix == ".pptx":
            # T6: PPTX 课件支持——zipfile 读 slide XML 提取 <a:t> 文本，按页拼接
            import zipfile
            import re as _re
            with zipfile.ZipFile(str(path)) as z:
                slides = sorted(
                    (x for x in z.namelist() if _re.match(r"ppt/slides/slide\d+\.xml$", x)),
                    key=lambda x: int(_re.findall(r"\d+", x)[0]),
                )
                parts = []
                for s in slides:
                    xml = z.read(s).decode("utf-8", "ignore")
                    texts = [t.strip() for t in _re.findall(r"<a:t>([^<]*)</a:t>", xml) if t.strip()]
                    parts.append("\n".join(texts))
                text = "\n\n".join(parts)
                if len(text.strip()) < config.PDF_MIN_CHARS_PER_PAGE:
                    print(f"[warn] 跳过无文字内容的 PPTX: {path.name}")
                    return None
                return text
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        if suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[warn] 跳过无法读取的 {path}: {e}")
    return None


def _split_recursive(text, chunk_size, separators):
    """纯递归分割（不合并）：返回按分隔符切开的文本片列表。

    与 _split_text 的区别：这里只切不 merge。修复背景（B 专项）：
    旧实现每层递归都调 _merge_splits，递归返回的『已合并块』又被父级当 split 再次合并，
    overlap 逐层叠加 → 同一文本块在单个 chunk 内重复（Q21 软链接块重复两遍即此所致），
    污染嵌入、破坏检索。现分割与合并彻底分离：分割阶段纯切，合并只在顶层做一次。
    """
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

    good = []
    for s in splits:
        if s == "":
            continue
        if len(s) < chunk_size:
            good.append(s)
        else:
            if len(separators) > 1:
                good.extend(_split_recursive(s, chunk_size, separators[1:]))
            else:
                good.append(s)
    return good


def _split_text(text, chunk_size, chunk_overlap, separators):
    """递归切分（模仿 RecursiveCharacterTextSplitter）：分割后只在顶层一次性合并。

    修复（B 专项）：原实现递归内部也 _merge_splits，overlap 多层叠加导致
    同一文本块在单 chunk 内重复。现 _split_recursive 只切不合并，_merge_splits
    只在外层调用一次，overlap 仅作用一次，杜绝块内重复。
    """
    return _merge_splits(_split_recursive(text, chunk_size, separators), chunk_size, chunk_overlap)


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


# ====================== 1.5 结构感知切分（T8，2026-08-22）======================
# 目标：代码块 / markdown 表格整体保护（不被 500 字符边界切断）；
#       按标题（# ~ ######）分层，每块 metadata 带 chapter（所属章节标题）。
# 实现：先把代码块/表格替换为哨兵串，切分后再还原；标题行作为节边界。
import re as _re


def _protect_blocks(text):
    """代码块(```...```)与连续表格行整体替换为哨兵，返回 (替换后文本, {哨兵: 原文本})。"""
    blocks = {}

    def _repl_code(m):
        key = f"§B{len(blocks)}§"
        blocks[key] = m.group(0)
        return key

    text = _re.sub(r"```.*?```", _repl_code, text, flags=_re.S)

    def _repl_table(m):
        key = f"§B{len(blocks)}§"
        blocks[key] = m.group(0)
        return key

    # 连续以 | 开头的行（markdown 表格）整体保护
    text = _re.sub(r"(?:^\|.*\|\s*$(\n|$))+", _repl_table, text, flags=_re.M)
    return text, blocks


def _restore_blocks(text, blocks):
    for k, v in blocks.items():
        text = text.replace(k, v)
    return text


def _split_structured(text, chunk_size, chunk_overlap, separators):
    """T8：结构感知切分。返回 [(chunk, start_index, chapter_or_None)]。
    - 代码块/表格整体保护（哨兵替换→切分→还原，永不切断）
    - 按标题分层：标题行作为节边界，块不跨章节；每节内走原 _split_with_index
    - 无标题文档退化为单节（chapter=None），行为与旧版一致
    """
    protected, blocks = _protect_blocks(text)
    lines = protected.split("\n")
    sections = []  # (chapter, section_start_offset, section_text)
    cur_title, cur_lines, cur_start = None, [], 0
    offset = 0
    for ln in lines:
        m = _re.match(r"^#{1,6}\s+(.+?)\s*$", ln.strip())
        if m:
            if cur_lines:
                sections.append((cur_title, cur_start, "\n".join(cur_lines)))
            cur_title = m.group(1).strip()
            cur_lines = [ln]
            cur_start = offset
        else:
            cur_lines.append(ln)
        offset += len(ln) + 1
    if cur_lines:
        sections.append((cur_title, cur_start, "\n".join(cur_lines)))

    out = []
    for chapter, sec_start, sec_text in sections:
        if not sec_text.strip():
            continue
        for chunk, rel_start in _split_with_index(sec_text, chunk_size, chunk_overlap, separators):
            out.append((_restore_blocks(chunk, blocks), sec_start + rel_start, chapter))
    return out


# ====================== 1.6 乱码清洗（2026-08-22）======================
# 微信小程序教材 PDF 的部分页面（章节小结/图表/课后练习）存在 CID 字体编码乱码
# （pypdf 提取出 阿拉伯/天城文/控制字符 等无意义字符，如 "֥ؓ ஥ഺb"）。
# 诊断：303 页 372 块中 29 块 >30% 乱码、251 块 >10%（广泛混入）。
# 处理：乱码占比 > GARBLED_DROP_THRESHOLD → 整块丢弃；否则剥离乱码字符保留正文。
# 注：剥离改变文本长度，start_index 仍保留原文偏移（引用定位容忍少量偏差）。

def _is_good_char(ch):
    code = ord(ch)
    # 2026-08-22 修复：原白名单含 0x00A0-0x00FF（拉丁补充区），本意是放行西欧字母；
    # 但 pypdf 对 CID 乱码的另一种映射就是产出 æ/ç/è 这类拉丁字符（实测 698 块中 54 块残留），
    # 被当作"好字符"放行 → 污染嵌入。修复：拉丁补充区不再整段放行，
    # 仅保留中文教材真实会用到的几个符号（· × ÷ ° 等）；真实西欧词句极少出现，损失可忽略。
    return (
        (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)  # CJK 基本+扩展A
        or (0x3000 <= code <= 0x303F) or (0xFF00 <= code <= 0xFFEF)  # CJK/全角标点
        or (0x20 <= code <= 0x7E) or (0x2010 <= code <= 0x206F)  # ASCII/通用标点
        or (0x00B7 == code or 0x00D7 == code or 0x00F7 == code or 0x00B0 == code)  # ·×÷°（中文教材常用）
        or (0x2018 <= code <= 0x201F)  # 引号
        or (0x2190 <= code <= 0x22FF)  # 箭头/数学运算符（教材公式）
    )


def clean_garbled_chunk(text):
    """乱码清洗：返回清洗后文本；返回 None 表示整块丢弃（乱码占比超阈值）。"""
    if not text:
        return text
    total = bad = 0
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(ch)
            continue
        total += 1
        if _is_good_char(ch):
            kept.append(ch)
        else:
            bad += 1
    ratio = bad / total if total else 0.0
    if ratio > config.GARBLED_DROP_THRESHOLD:
        return None
    return "".join(kept)


def load_and_chunk_documents(directory=config.DATA_DIR):
    # 支持 PDF / DOCX / TXT / MD；TXT/MD 强制 utf-8（Windows 中文坑，直接写死避免再踩）
    directory = Path(directory)
    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".pdf", ".txt", ".md", ".docx", ".pptx"):
            continue
        text = _load_text(path)
        if not text or not text.strip():
            continue
        for chunk, start, chapter in _split_structured(
            text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
        ):
            cleaned = clean_garbled_chunk(chunk)  # 乱码清洗：None=丢弃
            if cleaned is None:
                continue
            meta = {"source": str(path), "start_index": start}
            if chapter:
                meta["chapter"] = chapter  # T8: 所属章节标题（md 有标题时）
            documents.append(Document(page_content=cleaned, metadata=meta))
    print(f"Loaded and chunked {len(documents)} pieces from documents.")
    return documents


# ====================== 2. 嵌入 + 向量库（带缓存） ======================
class _OllamaEmbeddingFunction:
    """chromadb 1.5.x 兼容的嵌入函数：封装 ollama 的 nomic-embed-text。
    新式接口要求 __call__(self, input) 接受文本列表并返回 embedding 列表。"""

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._batch = 32  # 分批嵌入，避免大批量触发 Ollama llama-server 动态端口失效

    def name(self):
        # chromadb 1.5.x 要求嵌入函数提供 name()，用于持久化配置冲突校验
        return "ollama"

    def _embed_batch(self, batch):
        """单批嵌入 + 重试（Ollama 0.32.9 的 llama-server 动态端口在长任务中可能失效，
        11434 会 307 重定向到新端口；失败重试走新端口即可恢复）。"""
        import time as _time
        last_err = None
        for attempt in range(4):
            try:
                resp = self._client.embed(model=self._model, input=batch)
                return resp["embeddings"]
            except Exception as e:
                last_err = e
                _time.sleep(1.5 * (attempt + 1))
        raise last_err

    def __call__(self, input):
        # 建库时 chromadb 用此嵌入文档（input 为文本列表）；分批防止长任务中断
        items = list(input) if isinstance(input, (list, tuple)) else [input]
        out = []
        for i in range(0, len(items), self._batch):
            out.extend(self._embed_batch(items[i : i + self._batch]))
        return out

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
        # 2026-08-22 修复（系统测试 P1）：chunks 为空时 chromadb 的 upsert 会抛
        # "Non-empty lists are required" —— data/ 空目录重建索引时直接崩溃。
        # 空库也应返回有效集合句柄（count()==0），问答/健康检查照常工作。
        if chunks:
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
        print(f"Vector store created and saved ({len(chunks)} chunks).")
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
    for chunk, start, chapter in _split_structured(
        text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
    ):
        cleaned = clean_garbled_chunk(chunk)  # 乱码清洗：None=丢弃
        if cleaned is None:
            continue
        meta = {"source": str(path), "start_index": start}
        if chapter:
            meta["chapter"] = chapter
        docs.append(Document(page_content=cleaned, metadata=meta))
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
    T3（2026-08-22）：把 L2 distance 转 cosine 相似度写入 metadata['vector_score']，
    作为统一置信度信号（拒答/低置信阈值用）。
    """
    k = k or config.RETRIEVE_K
    collection = get_vector_store()
    res = collection.query(query_texts=[question], n_results=k)
    texts = res["documents"][0]
    metas = res["metadatas"][0]
    dists = (res.get("distances") or [[]])[0] if res.get("distances") else [None] * len(texts)
    docs = []
    for t, m, d in zip(texts, metas, dists):
        meta = dict(m) if m else {}
        if d is not None:
            # 归一化向量下 L2^2 = 2(1-cos)，故 cosine ≈ 1 - d^2/2（nomic-embed-text 归一化）
            meta["vector_score"] = round(1.0 - (d * d) / 2.0, 4)
        docs.append(Document(page_content=t, metadata=meta))
    return docs


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
    # B 专项：融合候选池扩到 RERANK_CANDIDATE_K（>RERANKER_TOP_N），
    # 让「仅 BM25 命中、向量缺席」的正确块先进池，再由 reranker 精排回 top-N（见 retrieve_and_rerank）。
    return _rrf_fuse([vector_docs, bm25_docs], top_n=config.RERANK_CANDIDATE_K)


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
        docs = hybrid_retrieve(question)  # 已融合出 RERANK_CANDIDATE_K 候选
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
        else:
            # 不开 reranker：候选池截回 RERANKER_TOP_N，保证进 LLM 的条数与旧版一致
            docs = docs[: config.RERANKER_TOP_N]
    else:
        docs = retrieve_vector(question, k=config.RETRIEVE_K)
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=config.RERANKER_TOP_N)
        else:
            docs = docs[: config.RERANKER_TOP_N]
    return docs


def get_confidence(docs):
    """T3（2026-08-22）：取检索结果 top-1 的向量相似度（cosine）作为置信度信号。
    各检索路径（快速/精确/hybrid/reranker）的 doc 都经过 retrieve_vector，vector_score 通常可得；
    hybrid 下若 top-1 是纯 BM25 召回块（无 vector_score）返回 None，调用方按「不提示」处理。"""
    if not docs:
        return None
    return docs[0].metadata.get("vector_score")


def is_hard_reject(docs):
    """P1（2026-08-24）：硬拒答检索阶段判定。

    返回 (reject: bool, signal: float)：
      - docs 为空 → 直接拒（检索无结果）。
      - top-1 向量相似度 < HARD_REJECT_VECTOR_THRESHOLD → 拒（明显语料外）。
      - 精确模式且 reranker 开启时，top-1 rerank_score < HARD_REJECT_RERANK_THRESHOLD → 拒。
    阈值经 eval_heldout.py（B 组拒答率应 ≥90%）校准；取 0.55 仅拦明显语料外，误拒率低。
    """
    if not docs:
        return True, 0.0
    top = docs[0]
    vs = top.metadata.get("vector_score")
    rs = top.metadata.get("rerank_score")
    if vs is not None and vs < config.HARD_REJECT_VECTOR_THRESHOLD:
        return True, vs
    if config.RERANKER_ENABLED and rs is not None and rs < config.HARD_REJECT_RERANK_THRESHOLD:
        return True, rs
    return False, (vs if vs is not None else (rs if rs is not None else 0.0))


# 硬拒答标准文案（语料外明确拒，而非软提示；符合系统提示「宁可少答不可错答」）。
HARD_REJECT_ANSWER = (
    "根据提供的课程资料无法回答该问题——检索到的内容与问题不相关（资料可能不足，"
    "或该问题超出本课程范围）。建议补充相关教材/笔记后重试，或换一种更贴近课程知识点的问法。"
)


def grounded_generate(question, docs, chain):
    """P1（2026-08-24）：带接地自检的生成。

    流程：先正常生成 → 若 GROUNDED_VERIFY_ENABLED，用独立裁判校验是否完全基于上下文；
    未接地则把答案替换为 HARD_REJECT_ANSWER 并返回 grounded=False。
    返回 (answer, grounded_flag, grounded_score, citations 不在此处理)。
    仅「精确模式」调用（保护快速模式延迟，见 app.py）。
    """
    answer = chain.invoke(question, docs=docs)
    grounded_flag = True
    grounded_score = 1.0
    if config.GROUNDED_VERIFY_ENABLED:
        context = "\n\n".join(d.page_content for d in docs)
        res = grounding.verify_grounded(context, answer)
        grounded_score = res["score"]
        if not res["grounded"]:
            answer = HARD_REJECT_ANSWER
            grounded_flag = False
    return answer, grounded_flag, grounded_score


# ====================== 4. 拼 RAG 链（含引用） ======================
SYSTEM_RULES = """你是一位严谨、准确的高校课程助教，专门解答专业课与课本问题。
请严格遵守以下规则：
1. 只能依据下面提供的【上下文】作答，严禁使用上下文之外的知识，也不得凭记忆编造。
   重要：只要【上下文】包含了与问题相关的信息，就必须基于上下文尽量答全、答准；即使某些细节在资料中未完整展开，也要把资料已给出的部分讲清楚，不要因"信息不完整"而拒答或只说"无法回答"。
2. 仅当【上下文】完全没有提供与问题相关的任何信息、或上下文内部相互矛盾时，才说明"根据提供的资料无法回答"。不要猜测，也不要因个别细节缺失就用通用知识编造补全。注意：即使上下文描述的是示例、虚构或假设性概念，只要其中已给出明确答案，就必须基于上下文如实作答，不得以此为由拒答。
3. 合理使用 Markdown（加粗、列表等）让回答清晰易读。
4. 回答语言必须与用户提问保持一致（用户用中文提问，就用中文回答）。
5. 回答要紧扣问题，避免堆砌与问题无关的内容。
6. 回答中每提及一条来自【上下文】的具体事实，必须紧跟来源标注 [n]（半角方括号，n 为下方『已知来源』中的编号，即上下文第 n 段）；同一句可标多个如 [1][2]。仅当某事实确实无法对应任何来源时才不标。严禁编造超出范围的序号（n 必须 ≤ 已知来源条数）。
   重要：回答正文中，凡来自【上下文】的事实句，句末/句首必须带至少一个 [n] 标注；整段回答不允许出现"无任何 [n] 标注的长段落"。若某句没有对应来源可标，说明该句不属于上下文内容，请删除或改写为"根据提供的资料无法确认"。"""


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


# ====================== 5. 统一问答入口（评审 P1，2026-08-30） ======================
def build_citations(docs):
    """检索结果 → 引用结构（UI / API / answer_question 三处共用，杜绝各写一份漂移）。

    分数展示优先级：rerank_score → hybrid_score → vector_score
    （快速模式只有 vector_score；混合未精排时是 hybrid_score；精排后是 rerank_score）。
    """
    citations = []
    for rank, d in enumerate(docs, 1):
        score = (
            d.metadata.get("rerank_score")
            or d.metadata.get("hybrid_score")
            or d.metadata.get("vector_score")
        )
        citations.append({
            "rank": rank,
            "source": Path(d.metadata.get("source", "unknown")).name,
            "content": d.page_content.strip(),
            "score": score,
            "start_index": d.metadata.get("start_index"),
        })
    return citations


def answer_question(question, mode="fast", custom_prompt=None, history="",
                    use_cache=True, on_stage=None):
    """统一问答入口：Streamlit / FastAPI / 评估三个出口共用同一套
    「缓存 → 检索 → 硬拒答 → 生成 → 接地自检 → 缓存写入」管线。

    背景（2026-08-30 外部评审 P1）：此前硬拒答与接地自检只挂在 app.py 的精确模式，
    main.py /query 明确绕过（「不做硬拒答/接地自检，非空即可缓存」）——held-out
    B 组 95% 拒答率只对 UI 成立，API 消费者拿到无闸门回答，且 API 缓存策略与 UI
    的「拒答不缓存」矛盾。现收敛为本函数，任何新增出口都必须走这里。

    mode 语义（与检索模式一致）：
      fast     纯向量 top-N + 软提示（low_confidence），无硬拒答/无接地自检（保护延迟）
      accurate 混合检索（+可选 reranker）+ 硬拒答 + 接地自检
    use_cache   语义缓存开关（评估侧必须传 False，避免吃到线上缓存污染结果）
    on_stage    可选回调 fn(stage)，stage ∈ {"retrieve","rerank","generate"}，
                供 UI 展示分阶段进度；缓存命中不触发。

    返回 dict：
      answer            最终答案（硬拒/未接地时已被替换为 HARD_REJECT_ANSWER）
      citations         [{rank,source,content,score,start_index}]（缓存命中来自缓存）
      docs              检索结果 List[Document]（缓存命中时 None）
      cached/cache_sim  是否命中语义缓存及相似度
      rejected/reject_signal                 检索阶段硬拒答及信号分
      grounding_ran/grounded/not_grounded    接地自检：是否执行/是否通过/是否未通过
      grounded_score    接地自检得分（未执行或检索拒时为 None）
      low_confidence    top-1 向量分 < LOW_CONFIDENCE_THRESHOLD（软提示，两模式都有）
    """
    stage = on_stage or (lambda s: None)
    mode_label = "精确" if mode == "accurate" else "快速"

    # 1) 语义缓存：命中跳过检索/重排/生成；写入策略（拒答/未接地不缓存）统一在下方 store
    if use_cache:
        cached = semantic_cache.lookup(question, mode_label)
        if cached:
            return {
                "answer": cached["answer"], "citations": cached["citations"],
                "docs": None, "cached": True, "cache_sim": cached["sim"],
                "rejected": False, "reject_signal": None,
                "grounding_ran": False, "grounded": True,
                "not_grounded": False, "grounded_score": None,
                "low_confidence": False,
            }

    # 2) 检索（accurate=混合+可选精排；fast=纯向量）
    stage("retrieve")
    docs = retrieve_and_rerank(question, force_no_rerank=(mode != "accurate"))
    if mode == "accurate" and config.RERANKER_ENABLED:
        stage("rerank")
    citations = build_citations(docs)

    # 3) 硬拒答（仅精确模式；快速模式保留软提示 low_confidence，不拦）
    rejected, reject_signal = False, None
    if mode == "accurate":
        rejected, reject_signal = is_hard_reject(docs)

    # 4) 生成 + 接地自检（仅精确模式）
    grounded, grounded_score, grounding_ran, not_grounded = True, None, False, False
    if rejected:
        answer = HARD_REJECT_ANSWER
        grounded = False
    else:
        stage("generate")
        rag = build_rag_chain(custom_prompt=custom_prompt, history=history)
        if mode == "accurate":
            answer, grounded, grounded_score = grounded_generate(question, docs, rag["chain"])
            grounding_ran = config.GROUNDED_VERIFY_ENABLED
            not_grounded = grounding_ran and not grounded
        else:
            answer = rag["chain"].invoke(question, docs=docs)

    # 5) 缓存写入：拒答（检索拒/未接地）不入缓存，与 UI 既有口径一致
    if use_cache:
        semantic_cache.store(question, mode_label, answer, citations,
                             reject=(rejected or not_grounded))

    conf = get_confidence(docs)
    return {
        "answer": answer, "citations": citations, "docs": docs,
        "cached": False, "cache_sim": None,
        "rejected": rejected, "reject_signal": reject_signal,
        "grounding_ran": grounding_ran, "grounded": grounded,
        "not_grounded": not_grounded, "grounded_score": grounded_score,
        "low_confidence": bool(conf is not None and conf < config.LOW_CONFIDENCE_THRESHOLD),
    }
