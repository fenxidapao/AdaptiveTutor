# -*- coding: utf-8 -*-
"""抽验：新 caption 是否真的进了向量库（含 metadata 完整性）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import rag_core

col = rag_core.get_vector_store(auto_rebuild=False)
res = col.get(where_document={"$contains": "tab1.wxml"}, include=["documents", "metadatas"])
print("含 tab1.wxml 的块数:", len(res["documents"]))
for d, m in zip(res["documents"], res["metadatas"]):
    m = m or {}
    print(f"--- image_path={m.get('image_path')} | type={m.get('type')} | page={m.get('page')}")
    print("   ", d[:130].replace("\n", " / "))
