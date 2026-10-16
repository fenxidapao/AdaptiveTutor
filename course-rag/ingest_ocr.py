"""
ingest_ocr.py — 教材接入（Gate 1，2026-08-24 修订）

把 E:\\书本 下「未入语料」的资料转成可索引 Markdown，写入 data/教材/<名>.md，
再由现有 load_and_chunk_documents 切分进库（run_ab.py / 上传即增量）。

智能探测策略（先讲难听的，再给方案）：
  - 很多「扫描书」未必是纯扫描——部分 PDF 自带文字层但 pypdf 能提取（直提）；
    部分含 Font/Tj 标记但无 Unicode 映射（「伪文字层」，如《算法设计与分析》），
    pypdf 提取为 0 字符，必须 OCR。2026-08-24 pypdf 实测：
      《离散数学（第2版）》413 页 0 字符 → 真扫描，OCR；
      《算法设计与分析第四版》358 页 0 字符（有标记无映射）→ 伪文字层，也 OCR；
      《扩展-远程客户端安装》22 页 941 字符 → 薄文字层，直提（量少）。
  - 因此本脚本**以 pypdf 实测为准**：能提取到足量字符就直提（快、准、零依赖）；
    只有真正 0 文字层才降级走 OCR 兜底。

OCR 兜底（本地优先，数据不出机，均懒加载，未安装不影响仓库编译/其他模块）：
  - ocrmypdf：扫描 PDF → 可搜索 PDF → 提取（最稳；依赖 ghostscript + tesseract，
    Windows 装系统依赖麻烦，本机未装，仅当已有时生效）。
  - RapidOCR + PyMuPDF（2026-09-03 起为正式兜底）：纯 pip 方案
    （`pip install rapidocr-onnxruntime pymupdf`），onnx 模型内置 wheel、离线可用，
    onnxruntime 后端与 rag_core 链（PyMuPDF/chromadb）同进程无 DLL 冲突（已冒烟实测）。
    实测（2026-09-03，RTX 4060 Laptop CPU 推理）：识别 3.4s/页 + 渲染 0.2s/页（200dpi），
    413 页全本约 0.4h；单栏正文近乎零错，双栏目录行序会乱（版面分析是已知边界）。
  - PaddleOCR：**已判死，勿再尝试**（2026-09-03 最小复现钉死：paddlepaddle 3.3.1 的
    Windows inference 层加载 PIR JSON 模型必抛 json parse_error.101，与 paddlex/
    设备检测/模型缓存均无关；且 paddle 原生 DLL 与 rag_core 链同进程 WinError 127）。
    判死证据链见 _archive/_ocr_pilot_standalone.py 头注释。
  - 云 API（TextIn xParse）仅在本地乱码不可接受时再考虑（文本出本机）。

原件一律保留不删（用户 2026-08-24 明确）——删除原始 PDF 不在本脚本职责内。

用法：
  python ingest_ocr.py                          # 处理 E:\\书本 下所有 PDF（跳过已在 data/教材 的重复文件）
  python ingest_ocr.py "E:\\书本\\<书>.pdf"      # 只处理一本
  python ingest_ocr.py "E:\\书本\\<书>.pdf" --max-pages 10   # 只 OCR 前 10 页（试点验证质量，再全量）
"""
import os
import sys
import argparse
from pathlib import Path

import config

OCR_OUT_DIR = Path(config.DATA_DIR) / "教材"
# 与 rag_core._load_text 一致的扫描判定阈值（每页均字符低于此值视为无文字层）
PDF_MIN_CHARS_PER_PAGE = int(getattr(config, "PDF_MIN_CHARS_PER_PAGE", 30))


def _pdf_has_text_layer(pdf_path: Path) -> bool:
    """判定是否有【可提取】文字层：直接 pypdf 实测。

    注意：不用字节探测（/Font /BT /Tj 标记）。实测发现两类 PDF 有标记但 pypdf 提取为空：
      1) 对象流压缩的 PDF（字体引用在压缩流里，原始字节不可见）；
      2) 「伪文字层」——含文本对象但无 Unicode 映射（ToUnicode CMap），extract_text 返回空，
         如 E:\\书本\\算法设计与分析第四版 (王晓东).pdf（358 页，Font/BT/Tj 齐全但 0 字符）。
    只有 pypdf 真能提出非空文本才算有文字层，否则一律走 OCR。
    """
    try:
        text = _extract_pdf_text(pdf_path)
    except Exception:
        return False
    return bool(text and text.strip())


def _extract_pdf_text(pdf_path: Path) -> str:
    """用 rag_core 的提取路径（PyMuPDF 优先、pypdf 兜底；复用扫描检测与告警逻辑）。

    2026-08-28 更正：原实现名为 _extract_via_pypdf 且文案称「pypdf 路径」——
    实测 pypdf 对 CID 乱码教材（如《微信小程序开发实战》）提取为乱码/空，
    而 PyMuPDF 可正常提取，故判定逻辑已随 rag_core 升级为 PyMuPDF 优先。"""
    from rag_core import _load_text
    return _load_text(Path(pdf_path)) or ""


def _is_scanned(text: str, pages_hint: int = 1) -> bool:
    """按 rag_core 口径判断是否扫描版：均字符/页 < 阈值。"""
    if not text.strip():
        return True
    # pages_hint 不可得时退化为总量阈值
    if pages_hint > 1 and (len(text.strip()) / pages_hint) < PDF_MIN_CHARS_PER_PAGE:
        return True
    if pages_hint <= 1 and len(text.strip()) < PDF_MIN_CHARS_PER_PAGE * 3:
        return True
    return False


def _ocr_ocrmypdf(pdf_path: Path, work_pdf: Path) -> Path:
    """OCR 优先路径：生成可搜索 PDF。"""
    from ocrmypdf import api as ocrmypdf_api
    ocrmypdf_api.ocr(
        str(pdf_path), str(work_pdf),
        lang="chi_sim+eng", skip_text=True, quiet=True,
    )
    return work_pdf


def _ocr_rapid(pdf_path: Path, max_pages: int | None = None) -> str:
    """RapidOCR 兜底（正式，2026-09-03 起）：PyMuPDF 逐页渲染 → onnxruntime 识别。

    纯 pip 依赖：`pip install rapidocr-onnxruntime pymupdf`；onnx 模型内置 wheel，
    首次运行无需联网。同进程安全：onnxruntime 与 rag_core 链（PyMuPDF/chromadb）
    无 DLL 符号冲突（2026-09-03 冒烟：import rag_core 后初始化+识别全通），
    故无需 subprocess 隔离（Paddle 路线才需要，已判死）。
    """
    try:
        import pymupdf  # PyMuPDF 3.x 推荐名（替代 fitz，消 deprecation warning）
    except ImportError:
        import fitz as pymupdf
    from rapidocr_onnxruntime import RapidOCR
    import tempfile

    ocr = RapidOCR()  # PP-OCRv4 det+cls+rec onnx，懒加载约 1s
    doc = pymupdf.open(str(pdf_path))
    total = len(doc)
    upto = total if max_pages is None else min(max_pages, total)
    print(f"[ocr-rapid] 渲染+识别前 {upto}/{total} 页（dpi=200）…")
    tmp_png = Path(tempfile.gettempdir()) / "_ingest_ocr_page.png"  # 逐页覆盖写
    lines = []
    for i in range(upto):
        pix = doc[i].get_pixmap(dpi=200)
        pix.save(str(tmp_png))
        try:
            result, elapse = ocr(str(tmp_png))
        except Exception as e:
            print(f"[ocr-rapid] 第 {i + 1} 页识别失败 {e!r}，跳过")
            continue
        if result:
            lines.extend(row[1] for row in result)  # row = [box, text, score]
        print(f"[ocr-rapid] 第 {i + 1}/{upto} 页：{len(result) if result else 0} 行")
    try:
        tmp_png.unlink(missing_ok=True)
    except OSError:
        pass  # 临时文件清理失败不影响主流程
    doc.close()
    return "\n".join(lines)


def ocr_to_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """兜底：无文字层时走 OCR（ocrmypdf 优先（若本机已装），RapidOCR 正式兜底）。

    max_pages 非 None（试点模式）：跳过 ocrmypdf（整本转换，不适合试点），直接 RapidOCR 前 N 页。
    """
    if max_pages is None:
        work = pdf_path.with_name(pdf_path.stem + "_searchable.pdf")
        try:
            _ocr_ocrmypdf(pdf_path, work)
            text = _extract_pdf_text(work)
            if text and text.strip():
                return text
        except Exception as e:
            print(f"[ocr] ocrmypdf 不可用/失败: {e!r}，转 RapidOCR 兜底…")
    else:
        print(f"[ocr] 试点模式（前 {max_pages} 页），直接走 RapidOCR…")
    return _ocr_rapid(pdf_path, max_pages=max_pages)


def ingest_one(pdf_path: Path, max_pages: int | None = None) -> Path:
    name = pdf_path.name
    print(f"[ingest] 处理: {name}")

    # 1) 智能分流：先探文字层，能直提就直提（快、准、零依赖）
    if _pdf_has_text_layer(pdf_path):
        text = _extract_pdf_text(pdf_path)
        if not _is_scanned(text):
            print(f"[ingest] 检测到文字层，直接提取（免 OCR）")
            OCR_OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OCR_OUT_DIR / f"{pdf_path.stem}.md"
            out.write_text(
                f"# {pdf_path.stem}（直提录入）\n\n> 来源：{name}（含文字层，pypdf 直接提取）\n\n{text}\n",
                encoding="utf-8",
            )
            print(f"[ingest] 写出 {out}（{len(text)} 字）→ 随后跑 `python run_ab.py` 重建索引")
            return out
        else:
            print(f"[ingest] 文字层过薄（疑似扫描版），降级 OCR…")

    # 2) 兜底：OCR
    print(f"[ingest] OCR: {name}")
    text = ocr_to_text(pdf_path, max_pages=max_pages)
    if not text or not text.strip():
        raise RuntimeError("OCR 输出为空，可能依赖未装或 PDF 损坏")
    OCR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_前{max_pages}页试点" if max_pages else ""
    out = OCR_OUT_DIR / f"{pdf_path.stem}_OCR{suffix}.md"
    out.write_text(f"# {pdf_path.stem}（OCR 录入{suffix}）\n\n{text}\n", encoding="utf-8")
    print(f"[ingest] 写出 {out}（{len(text)} 字）→ 随后跑 `python run_ab.py` 重建索引")
    return out


def _already_in_corpus(pdf_path: Path) -> bool:
    """跳过 data/教材 里已存在的同名/同内容 PDF（避免重复入语料）。"""
    cand = OCR_OUT_DIR / f"{pdf_path.stem}.md"
    cand2 = OCR_OUT_DIR / f"{pdf_path.stem}_OCR.md"
    return cand.exists() or cand2.exists()


def main():
    ap = argparse.ArgumentParser(description="教材接入（Gate 1，文字层直提优先 + OCR 兜底）")
    ap.add_argument("path", nargs="?", default=None, help="PDF 文件或目录；默认处理 E:\\书本")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="只 OCR 前 N 页（试点验证质量，再全量跑）")
    args = ap.parse_args()
    root = Path(args.path) if args.path else Path(r"data/教材")
    if not root.exists():
        print(f"路径不存在: {root}")
        sys.exit(1)
    pdfs = [root] if root.is_file() else sorted(root.rglob("*.pdf"))
    if not pdfs:
        print("未发现 PDF")
        sys.exit(1)
    print(f"待处理: {len(pdfs)} 本（来自 {root}）")
    for p in pdfs:
        try:
            if _already_in_corpus(p):
                print(f"[skip] {p.name}: 已在 data/教材 存在，跳过（避免重复入语料）")
                continue
            ingest_one(p, max_pages=args.max_pages)
        except Exception as e:
            print(f"[skip] {p.name}: {e!r}")


if __name__ == "__main__":
    main()
