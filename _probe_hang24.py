# -*- coding: utf-8 -*-
"""定位「第 24 题检索卡住」：45 秒后打印栈回溯，看它到底卡在哪一行。

用法：python _probe_hang24.py
"""
import faulthandler
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("SKIP_REBUILD", "1")
os.environ.setdefault("VISUAL_RERANK_ENABLED", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 45 秒后无条件打印所有线程栈（然后继续跑）
faulthandler.dump_traceback_later(45, repeat=False)

import config  # noqa: E402

print(f"PERSIST_DIRECTORY = {config.PERSIST_DIRECTORY}", flush=True)
print(f"RERANKER_ENABLED  = {config.RERANKER_ENABLED}", flush=True)
print(f"HYBRID_ENABLED    = {config.HYBRID_ENABLED}", flush=True)

import rag_core  # noqa: E402
from eval_data_test import TEST_SET  # noqa: E402

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 23
q = TEST_SET[idx]["question"]
print(f"\n=== TEST_SET[{idx}] = {q}\n", flush=True)

t0 = time.time()
docs = rag_core.retrieve_and_rerank(q)
print(f"retrieve_and_rerank 完成：{len(docs)} 条，用时 {time.time()-t0:.1f}s", flush=True)
for i, d in enumerate(docs[:8], 1):
    md = d.metadata or {}
    print(f"  {i}. type={md.get('type') or 'text':5s} len={len(d.page_content):6d} "
          f"src={os.path.basename(str(md.get('source') or ''))[:34]}", flush=True)
