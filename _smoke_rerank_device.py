# -*- coding: utf-8 -*-
"""_smoke_rerank_device.py — reranker 设备冒烟测试（2026-09-03 GPU 化改造后的一次性验证）

验证三件事：
  1. 当前 venv 的 torch 是哪版轮子、CUDA 是否可用；
  2. reranker 实际落在哪个设备、权重加载耗时；
  3. 一次 16 条（~800 字符真实教材块）pair 打分耗时——模拟精确模式
     RERANK_CANDIDATE_K=16 的单次精排，对照 CPU 官方口径（单次 13~44s）。

用法：
  python _smoke_rerank_device.py            # 走 config.RERANKER_DEVICE（默认 auto）
  RERANKER_DEVICE=cpu python _smoke_rerank_device.py   # 强制 CPU 对照
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 权重已缓存，防 huggingface.co 不可达卡死
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
else:
    print("!! 未检测到 CUDA：若装的是 cu126 轮子仍为 False，多半是显卡驱动过旧")

import config
from reranker import get_reranker, rerank_documents

t0 = time.time()
rr = get_reranker()
print(f"[1/2] 权重加载: {time.time() - t0:.1f}s")

# 取真实教材正文切 16 条 ~800 字符块（零模型依赖，只读 data/教材 下第一个 .md）
docs = []
for name in sorted(os.listdir(config.DATA_DIR)):
    p = os.path.join(config.DATA_DIR, name)
    if os.path.isdir(p):
        for sub in sorted(os.listdir(p)):
            if sub.endswith(".md"):
                p = os.path.join(p, sub)
                break
    if p.endswith(".md"):
        break
with open(p, encoding="utf-8") as f:
    text = f.read()
CHUNK = 800
for i in range(16):
    chunk = text[i * CHUNK:(i + 1) * CHUNK]
    if len(chunk) < 400:  # 末尾不足则回绕拼接，保证长度贴近真实 chunk
        chunk = (chunk + text)[:CHUNK]
    docs.append(type("D", (), {"page_content": chunk, "metadata": {}})())
print(f"[语料] 用 {os.path.relpath(p)} 切出 {len(docs)} 条块（均 {CHUNK} 字符）")

q = "什么是事务的 ACID 特性"
t1 = time.time()
out = rerank_documents(q, docs, top_n=config.RERANKER_TOP_N)
dt = time.time() - t1
print(f"[2/2] 16-pair 打分: {dt:.2f}s  (top-{len(out)}，top1 rerank_score={out[0].metadata['rerank_score']:.4f})")
print(f"\n对照：CPU 官方口径单次精排 13~44s（EVALUATION.md L81，n=10）")
