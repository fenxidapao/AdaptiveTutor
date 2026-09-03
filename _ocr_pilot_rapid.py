# -*- coding: utf-8 -*-
"""_ocr_pilot_rapid.py — 扫描版 OCR 试点（RapidOCR 引擎版，2026-09-03）

背景（2026-09-03 排障结论）：
  PaddleOCR 3.7/paddlex 3.7.2 路线死因已钉死——paddlepaddle 3.3.1 的
  paddle_inference 在 Windows 上加载 PIR JSON 模型（inference.json）时
  create_predictor 抛 json.exception.parse_error.101，最小复现（绕过 paddlex
  直调 Config+create_predictor）同样崩 → 是 paddlepaddle 自身 bug，非参数问题。
  按工程判断切 RapidOCR（onnxruntime 后端，PP-OCRv4 模型的 onnx 移植）完成试点。

产出：
  - 逐页耗时（渲染/识别分开计时）→ 推全量 413 页成本
  - OCR 文本质量样本 → data/教材/<stem>_OCR_前N页试点_RapidOCR.md
用法：
  python _ocr_pilot_rapid.py --pdf "E:\\书本\\离散数学（第2版）....pdf" --pages 10
"""
import argparse
import time
from pathlib import Path

t0 = time.time()
from rapidocr_onnxruntime import RapidOCR  # onnxruntime 后端，自包含无 paddle DLL

t_rapid = time.time() - t0
print(f"[t] rapidocr import: {t_rapid:.1f}s", flush=True)

import pymupdf  # 渲染引擎后加载

ap = argparse.ArgumentParser(description="扫描版 OCR 试点（RapidOCR 独立进程）")
ap.add_argument("--pdf", required=True)
ap.add_argument("--pages", type=int, default=10)
ap.add_argument("--out-dir", default=None, help="输出目录（默认 data/教材）")
args = ap.parse_args()

pdf_path = Path(args.pdf)
t1 = time.time()
ocr = RapidOCR()  # 默认 PP-OCRv4 det+cls+rec onnx 模型
t_init = time.time() - t1
print(f"[t] RapidOCR 初始化(含模型加载): {t_init:.1f}s", flush=True)

doc = pymupdf.open(str(pdf_path))
upto = min(args.pages, len(doc))
print(f"[t] PDF 打开: {pdf_path.name}，共 {len(doc)} 页，本次试点前 {upto} 页", flush=True)

lines, page_secs, render_secs = [], [], []
img = Path("_ocr_tmp_page_rapid.png")
for i in range(upto):
    ta = time.time()
    pix = doc[i].get_pixmap(dpi=200)
    pix.save(str(img))
    tb = time.time()
    result, elapse = ocr(str(img))
    tc = time.time()
    texts = []
    if result:
        texts = [row[1] for row in result]  # [box, text, score]
    page_text = "\n".join(texts)
    lines.append(f"<!-- 第 {i+1} 页 -->\n{page_text}")
    page_secs.append(tc - tb)
    render_secs.append(tb - ta)
    print(f"[p{i+1:>2}/{upto}] 渲染 {tb-ta:.1f}s + 识别 {tc-tb:.1f}s，{len(texts)} 行 {len(page_text)} 字", flush=True)
try:
    img.unlink(missing_ok=True)  # 临时 PNG 清理（沙箱安全删除策略可能拦截，不影响主流程）
except OSError as e:
    print(f"[warn] 临时文件 {img} 清理失败（不影响结果）: {e}", flush=True)
total_pages = len(doc)
doc.close()

total = time.time() - t0
out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "data" / "教材"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / f"{pdf_path.stem}_OCR_前{upto}页试点_RapidOCR.md"
out.write_text(
    f"# {pdf_path.stem}（OCR 录入_前{upto}页试点，RapidOCR 1.2.3 / onnxruntime / PP-OCRv4）\n\n"
    f"> 试点数据：{upto} 页合计识别 {sum(page_secs):.1f}s（均 {sum(page_secs)/upto:.1f}s/页），"
    f"渲染 {sum(render_secs):.1f}s（均 {sum(render_secs)/upto:.1f}s/页，200dpi）；"
    f"全本 {total_pages} 页按此速率约 {sum(page_secs)/upto*total_pages/3600:.1f} 小时（CPU）。\n\n"
    + "\n\n".join(lines) + "\n",
    encoding="utf-8",
)
print(f"\n== 试点汇总 ==", flush=True)
print(f"识别: {sum(page_secs):.1f}s（均 {sum(page_secs)/upto:.1f}s/页）", flush=True)
print(f"渲染: {sum(render_secs):.1f}s（均 {sum(render_secs)/upto:.1f}s/页）", flush=True)
print(f"全本 {total_pages} 页 CPU 估算: {sum(page_secs)/upto*total_pages/3600:.1f} 小时", flush=True)
print(f"输出: {out}（{out.stat().st_size} 字节）", flush=True)
print(f"总耗时(含初始化): {total:.1f}s", flush=True)
