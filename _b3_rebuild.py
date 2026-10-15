# -*- coding: utf-8 -*-
"""B3 步 3：重建索引（让新 caption 真正进向量库）。

为什么必须重建：图像块在库里的**全部可检索内容**就是 caption
（`rag_core.load_image_chunks`：`page_content = caption`）。换了 caption 文件但
不重建，库里的图像块还是旧文字 —— 评测数字不会动。

重建会**删掉整个 collection 再重新嵌入**（`rag_core.get_vector_store`），所以：
  - 文本块（1219）也会被重新嵌入一次（内容没变，向量理论上一致）
  - 总块数应仍是 1434 = 1219 文本 + 215 图像
  - 想复跑换描述器之前的基线：设 IMAGE_CAPTIONS_CSV=_image_captions_v1_qwen25vl.csv 再跑本脚本

用法（cwd 必须是 CourseRAG 根目录）：
    python _b3_rebuild.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402
import rag_core  # noqa: E402


def main():
    print(f"IMAGE_CAPTIONS_CSV = {config.IMAGE_CAPTIONS_CSV}")
    cap = pathlib.Path(config.DATA_DIR) / "images" / config.IMAGE_CAPTIONS_CSV
    print(f"caption 文件        = {cap}  存在={cap.is_file()}")
    if not cap.is_file():
        raise SystemExit("caption 文件不存在，先跑 _caption_images_v2.py")

    t0 = time.time()
    chunks = rag_core.load_and_chunk_documents()
    n_img = sum(1 for d in chunks if d.metadata.get("type") == "image")
    print(f"载入 {len(chunks)} 块（其中图像块 {n_img}）  用时 {time.time() - t0:.1f}s")

    t1 = time.time()
    rag_core.get_vector_store(chunks=chunks)
    print(f"重建完成  用时 {time.time() - t1:.1f}s")

    col = rag_core.get_vector_store()
    print(f"库内块数 = {col.count()}")
    got = col.get(where={"type": "image"}, include=["metadatas"])
    print(f"库内图像块 = {len(got['metadatas'])}")

    # 抽验：新 caption 是否真的进库（拿判别性 token 当探针）
    probe = "tab1.wxml"
    res = col.get(where_document={"$contains": probe}, include=["metadatas"])
    print(f"库内含「{probe}」的块数 = {len(res['metadatas'])} (旧基线应为 0)")
    for m in res["metadatas"][:5]:
        print(f"   - {m.get('image_path')}")
    print(f"\n总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
