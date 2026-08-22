"""
main.py — CourseRAG FastAPI 服务入口（部署层，2026-08-20 新增）。

把 rag_core 的「索引构建 / 检索 / 重排 / 生成」能力封装为 RESTful 端点，
与 Streamlit（app.py）共用同一套核心模块，无重复逻辑：
  - 问答        POST /query              检索 + 重排 + 生成（含引用溯源）
  - 检索        POST /retrieve           仅检索 + 重排，不生成（供前端 / 调试）
  - 文档        GET/POST /documents      列出 / 上传（保存 + 增量索引）
  - 文档删除    DELETE /documents/{name} 删文件 + 从索引移除
  - 索引        POST /index/rebuild      全量重建（data/ 全部）
  - 统计        GET /index/stats         向量库块数 / 源文档数
  - 健康检查    GET /health              依赖 / Ollama / 索引自检

启动：uvicorn main:app --host 0.0.0.0 --port 8000（交互文档 http://localhost:8000/docs）

注意（与 app.py 相同的时序坑）：reranker 权重加载需访问 HuggingFace，本机
huggingface.co 不可达会超时。必须在 import 任何可能拉起 sentence_transformers /
huggingface_hub 的模块之前设置 HF_ENDPOINT——这里在 import rag_core 之前设置。
"""
import os

# 默认走镜像；若本机已缓存权重则强制离线，彻底不碰网络（与 app.py 一致）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_rerank_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
)
if os.path.isdir(_rerank_cache):
    os.environ["HF_HUB_OFFLINE"] = "1"

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from api_schemas import (
    CheckItem,
    Citation,
    DeleteResponse,
    DocInfo,
    DocumentsResponse,
    HealthResponse,
    IndexStats,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RebuildResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from health import collect_status
from query_log import log_query
from rag_core import (
    add_documents,
    build_rag_chain,
    chunk_file,
    delete_by_source,
    get_vector_store,
    load_and_chunk_documents,
    retrieve_and_rerank,
)

# 允许的上传扩展名（与 rag_core._load_text 支持的后缀保持一致）
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".pptx"}


# ==================== 通用工具 ====================
def _build_citations(docs) -> list[Citation]:
    """把检索到的 Document 列表转成与 app.py 完全一致的引用结构。"""
    citations = []
    for rank, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        content = doc.page_content.strip()
        score = doc.metadata.get("rerank_score") or doc.metadata.get("hybrid_score")
        start = doc.metadata.get("start_index")
        citations.append(
            Citation(
                rank=rank,
                source=source,
                content=content,
                score=float(score) if score is not None else None,
                start_index=start,
            )
        )
    return citations


def _retrieve(question: str, mode: str):
    """统一检索入口：fast=纯向量 top-N；accurate=向量+BM25（+reranker）。"""
    return retrieve_and_rerank(question, force_no_rerank=(mode != "accurate"))


def _safe_filename(name: str) -> str:
    """防路径穿越：只允许纯文件名，拒绝 .. / 分隔符。"""
    if not name or name != Path(name).name or name in (".", ".."):
        raise HTTPException(status_code=400, detail=f"非法文件名: {name!r}")
    return name


def _load_index_or_raise():
    """确保向量库可用；Ollama 未启动 / 库损坏时给出可操作的报错。"""
    try:
        return get_vector_store()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"向量库不可用：{e}。请确认 Ollama 已启动（ollama serve）且模型已拉取。",
        )


# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动预热：懒加载向量库（库为空且有资料会自动重建）。失败不阻塞启动——
    # Ollama 没启动时健康检查会明确报出来，而不是让服务起不来。
    try:
        get_vector_store()
        print("[api] vector store ready")
    except Exception as e:
        print(f"[api] warmup skipped (service will still start): {e}")
    yield


app = FastAPI(
    title="CourseRAG API",
    description="专业课 / 课本问答 RAG 服务。检索 + 重排 + 生成，来源可溯源。"
    "交互式文档见 /docs（Swagger UI）。",
    version="1.0.0",
    lifespan=lifespan,
)

# 演示 / 前后端分离场景放开跨域；生产可收紧为具体域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 服务信息 ====================
@app.get("/", tags=["meta"])
def service_info():
    """服务元信息（端点一览）。"""
    return {
        "name": "CourseRAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "POST /query",
            "POST /retrieve",
            "GET  /documents",
            "POST /documents",
            "DELETE /documents/{name}",
            "POST /index/rebuild",
            "GET  /index/stats",
        ],
        "models": {"embedding": config.EMBEDDING_MODEL, "llm": config.LLM_MODEL},
        "ollama": config.OLLAMA_BASE_URL,
    }


# ==================== 健康检查 ====================
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """依赖 / Ollama / Reranker 权重自检（复用 health.py），附带向量库块数。

    依赖清单按 API 视角检查（核心栈即可，无需 streamlit / sentence-transformers——
    镜像内 reranker 默认关，不装 torch 是预期行为）。
    """
    checks = collect_status(deps=["ollama", "chromadb", "pypdf", "docx", "dotenv"])
    count = None
    try:
        count = get_vector_store().count()
    except Exception:
        pass
    return HealthResponse(
        ok=all(ok for _, ok, _ in checks),
        checks=[CheckItem(label=l, ok=ok, msg=m) for l, ok, m in checks],
        vector_store_count=count,
    )


# ==================== 问答 / 检索 ====================
@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    """核心问答：检索 + 重排 + 生成，返回带 [n] 内联引用的回答与来源列表。

    与 Streamlit 完全同源：同一份 docs 既用于引用展示、又喂给 LLM，
    杜绝『展示的来源 ≠ 生成的来源』。响应含 latency_ms 便于前端展示耗时。
    """
    t0 = time.time()
    try:
        docs = _retrieve(req.question, req.mode)
        citations = _build_citations(docs)

        rag = build_rag_chain(custom_prompt=req.custom_prompt, history=req.history or "")
        answer = rag["chain"].invoke(req.question, docs=docs)

        log_query(
            question=req.question,
            mode="精确" if req.mode == "accurate" else "快速",
            citations=[c.model_dump() for c in citations],
            answer=answer,
        )
        # grounded 判定与 query_log._is_grounded 的启发式保持一致：
        # 非空 且 未声称「无法回答」→ 视为接地
        grounded = bool(answer.strip()) and "无法回答" not in answer
        return QueryResponse(
            answer=answer,
            mode=req.mode,
            citations=citations,
            latency_ms=int((time.time() - t0) * 1000),
            grounded=grounded,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成失败：{e}。请确认 Ollama 已启动（ollama serve）且模型已拉取（ollama pull {config.LLM_MODEL}）。",
        )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve(req: RetrieveRequest):
    """仅检索 + 重排（不调用 LLM），返回来源列表——供前端预检索 / 调试 / 评估。"""
    try:
        docs = _retrieve(req.question, req.mode)
        return RetrieveResponse(query=req.question, mode=req.mode, docs=_build_citations(docs))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")


# ==================== 文档管理 ====================
@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents():
    """列出 data/ 下所有文档，并给出各自在向量库中的块数（0=尚未索引）。"""
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    files, total = [], 0
    try:
        col = get_vector_store()
        # 按 source 统计每个文档的块数（metadata.source 为绝对/相对路径）
        all_meta = col.get(include=["metadatas"])["metadatas"]
        from collections import Counter

        chunk_counter = Counter(m.get("source") for m in all_meta if m)
    except Exception:
        chunk_counter = {}

    for p in sorted(data_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            files.append(
                DocInfo(
                    name=p.name,
                    size_bytes=p.stat().st_size,
                    chunks=chunk_counter.get(str(p), 0),
                )
            )
            total += chunk_counter.get(str(p), 0)
    return DocumentsResponse(files=files, total_chunks=total)


@app.post("/documents", response_model=IngestResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(..., description="PDF / TXT / MD / DOCX 文件")):
    """上传课程资料：保存到 data/ 并增量索引（稳定 id 去重，重复上传不产生重复块）。

    首次建库（向量库为空）时自动退化为从 data/ 全量重建，避免已有资料被漏索引——
    与 app.py 侧边栏的首次建库行为保持一致。
    """
    name = _safe_filename(file.filename or "")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix!r}，仅支持 {sorted(ALLOWED_SUFFIXES)}",
        )

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")
    dest.write_bytes(content)

    try:
        if get_vector_store().count() == 0:
            # 首次建库：全量重建（与 app.py 逻辑一致）
            chunks = load_and_chunk_documents()
            get_vector_store(chunks)
            added = len(chunks)
            print(f"[api] first index: rebuilt {added} chunks from data/")
        else:
            added = add_documents(chunk_file(dest))
        return IngestResponse(file=name, saved_to=str(dest), added_chunks=added)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引失败：{e}")


@app.delete("/documents/{name}", response_model=DeleteResponse, tags=["documents"])
def delete_document(name: str):
    """删除文档：移除 data/ 下的文件，并只从索引中移除该 source 的块（其余文档不受影响）。

    健壮性：文件删除失败（Windows 上被占用 / 权限不足）时不 500——
    索引照常移除，响应带 warning 提示，保证『文档内容已不可检索』是确定性的。
    """
    safe = _safe_filename(name)
    data_dir = Path(config.DATA_DIR)
    path = data_dir / safe

    warning = None
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            warning = f"文件删除失败（可能被占用或权限不足）：{e}；已从索引移除该文档。"

    # 无论文件在不在，都尝试从索引移除（防止文件已被手动删除但索引残留）
    try:
        delete_by_source(str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除索引失败：{e}")
    return DeleteResponse(deleted=True, filename=safe, warning=warning)


# ==================== 索引管理 ====================
@app.post("/index/rebuild", response_model=RebuildResponse, tags=["index"])
def rebuild_index():
    """全量重建索引：重新加载 data/ 全部文档并重建向量库（慢操作，适合初次 / 换语料后）。"""
    chunks = load_and_chunk_documents()
    if not chunks:
        raise HTTPException(status_code=400, detail="data/ 下没有可索引的文档（PDF / TXT / MD / DOCX）")
    get_vector_store(chunks)
    return RebuildResponse(chunks=len(chunks), files=len({d.metadata["source"] for d in chunks}))


@app.get("/index/stats", response_model=IndexStats, tags=["index"])
def index_stats():
    """向量库统计：总块数、已索引源文档数、持久化目录。"""
    col = _load_index_or_raise()
    metas = col.get(include=["metadatas"])["metadatas"]
    source_files = len({m.get("source") for m in metas if m})
    return IndexStats(
        collection=col.name,
        count=col.count(),
        source_files=source_files,
        persist_directory=str(Path(config.PERSIST_DIRECTORY).resolve()),
    )
