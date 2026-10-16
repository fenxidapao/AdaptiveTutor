# -*- coding: utf-8 -*-
"""P2 步1：教材 PDF 图像提取（bbox 渲染剪裁路线，P0 探测决策 2026-09-05）。

为什么用 bbox 渲染剪裁而非 extract_image（P0 实测决策）：
  1. 小程序书系统性反色（黑底白字）——渲染剪裁按所见即所得，自动修复；
  2. 渲染自动合成位图+矢量分层；
  3. 分辨率按原始像素反推 zoom（上限 3.0≈216dpi），caption/展示均够用。

过滤与合并规则（P0 决策报告 §3）：
  - 跳过封面/封底页（各 1 页）；
  - 退化/越界 bbox 跳过（ISBN 条码 bbox 越界为负，天然被滤）；
  - 原始 min 边 <200px 跳过（图标/装饰）；
  - 同页相邻图对象合并（一张逻辑图被拆成多 xref 的场景），min(dx,dy)<20pt 且 max<80pt；
  - 内容级去重（PNG bytes md5，治页眉横幅/同图多实例）。

产物（PNG/CSV 不会被 load_and_chunk_documents 吃进语料——只认 pdf/txt/md/docx/pptx）：
  data/images/<书名>/p<页码>_i<序>.png
  data/images/_manifest.csv（caption 步骤与展示步骤的清单依据）
"""
import pymupdf
import os
import csv
import hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "教材")
OUT = os.path.join(ROOT, "data", "images")
os.makedirs(OUT, exist_ok=True)

MIN_EDGE_PX = 200          # 原始分辨率 min 边阈值
MERGE_GAP_PT = 20          # 同页相邻对象合并：单方向贴近阈值
MERGE_MAX_PT = 80          # 另一方向最大间距（防跨图误并）
ZOOM_CAP = 3.0             # 渲染 zoom 上限（≈216dpi）
SKIP_PAGES = {1}           # 封面；封底在循环里按页数动态加


def _gap(a, b):
    """两个 rect 的 (dx, dy) 间距（不相交为正）。"""
    dx = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return dx, dy


def _merge_boxes(items):
    """同页相邻图对象贪心合并。items: [(rect, raw_w, raw_h, xref)] -> [合并组]。"""
    groups = []
    for rect, rw, rh, xref in sorted(items, key=lambda t: (t[0].y0, t[0].x0)):
        placed = False
        for g in groups:
            dx, dy = _gap(g["rect"], rect)
            if min(dx, dy) < MERGE_GAP_PT and max(dx, dy) < MERGE_MAX_PT:
                g["rect"] = pymupdf.Rect(
                    min(g["rect"].x0, rect.x0), min(g["rect"].y0, rect.y0),
                    max(g["rect"].x1, rect.x1), max(g["rect"].y1, rect.y1),
                )
                g["raw_w"] = max(g["raw_w"], rw)
                g["raw_h"] = max(g["raw_h"], rh)
                g["xrefs"].append(xref)
                placed = True
                break
        if not placed:
            groups.append({"rect": pymupdf.Rect(rect), "raw_w": rw, "raw_h": rh, "xrefs": [xref]})
    return groups


rows = []
seen_hash = {}
pdfs = sorted(f for f in os.listdir(DATA) if f.lower().endswith(".pdf"))
print(f"PDFs: {len(pdfs)}", flush=True)

for name in pdfs:
    book = os.path.splitext(name)[0]
    book_dir = os.path.join(OUT, book)
    os.makedirs(book_dir, exist_ok=True)
    doc = pymupdf.open(os.path.join(DATA, name))
    n_pages = len(doc)
    skip = SKIP_PAGES | {n_pages}
    n_kept, n_dup = 0, 0
    for pno in range(n_pages):
        page_no = pno + 1
        if page_no in skip:
            continue
        page = doc[pno]
        prect = page.rect
        items = []
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref") or 0
            bbox = info.get("bbox")
            rw, rh = int(info.get("width") or 0), int(info.get("height") or 0)
            if xref <= 0 or not bbox or rw == 0 or rh == 0:
                continue
            if min(rw, rh) < MIN_EDGE_PX:
                continue
            try:
                rect = pymupdf.Rect(bbox) & prect
            except Exception:
                continue
            if rect.is_empty or rect.width < 10 or rect.height < 10:
                continue
            items.append((rect, rw, rh, xref))
        if not items:
            continue
        for gi, g in enumerate(_merge_boxes(items)):
            clip = g["rect"] & prect
            zoom = max(1.0, min(g["raw_w"] / max(1.0, clip.width), ZOOM_CAP))
            try:
                pix = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(zoom, zoom))
            except Exception as e:
                print(f"[warn] {book} p{page_no} g{gi} render fail: {e}", flush=True)
                continue
            png = pix.tobytes("png")
            h = hashlib.md5(png).hexdigest()
            if h in seen_hash:
                n_dup += 1
                continue
            seen_hash[h] = f"{book} p{page_no}"
            fn = f"p{page_no:03d}_i{gi:02d}.png"
            path = os.path.join(book_dir, fn)
            with open(path, "wb") as f:
                f.write(png)
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            rows.append({
                "book": book, "page": page_no, "xrefs": ";".join(map(str, g["xrefs"])),
                "raw_w": g["raw_w"], "raw_h": g["raw_h"],
                "out_w": pix.width, "out_h": pix.height,
                "png_path": rel, "md5": h,
            })
            n_kept += 1
    doc.close()
    print(f"{book}: kept={n_kept} dup={n_dup}", flush=True)

csv_path = os.path.join(OUT, "_manifest.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                       ["book", "page", "xrefs", "raw_w", "raw_h", "out_w", "out_h", "png_path", "md5"])
    w.writeheader()
    w.writerows(rows)
print(f"\nTOTAL extracted={len(rows)} -> {csv_path}", flush=True)
