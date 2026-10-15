# -*- coding: utf-8 -*-
"""探针：证据门的 CJK 盲区 —— 字符级 token 让「个人信息」这类词永远当不了证据词。

## 问题

`visual_rerank._content_tokens` 用 `rag_search_utils._tokenize`（中文**逐字**切分）+
语料 DF 过滤。于是「个人信息」→ 个/人/信/息 四个**常见字**，DF 全部超高 → 被当噪声丢掉，
证据门对**任何中文专有名词**都判「无交集」。

实测（`_b1_bench_after_b3.txt`）：VLM 正确命中了 `p045_i00.png`，证据原文以
「个人信息, images, avatar.png, …」开头，却仍被丢弃 —— 正确命中被自己的门挡掉。

## 本探针要回答

1. jieba 分词后，「个人信息」「比较数字大小」「用户登录」「电影列表」的 DF 是多少？
   是否低于 `_EVIDENCE_MAX_DF`？
2. 字符级 vs 词级，判别性词的 DF 对比。

用法（cwd 必须是 CourseRAG 根目录）：
    python _probe_df_jieba.py
"""
import csv
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
import visual_rerank as vr
from rag_search_utils import _tokenize

CAP = pathlib.Path(config.DATA_DIR) / "images" / config.IMAGE_CAPTIONS_CSV
print(f"caption 语料 = {CAP}  存在={CAP.is_file()}")

try:
    import jieba
    print(f"jieba 可用，版本 {getattr(jieba, '__version__', '?')}")
except Exception as e:
    print(f"jieba 不可用：{e}")
    raise SystemExit(1)

words, chars = Counter(), Counter()
n = 0
with open(CAP, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        cap = row.get("caption", "") or ""
        n += 1
        for w in set(jieba.cut(cap)):
            w = w.strip()
            if len(w) >= 2:
                words[w] += 1
        for c in set(_tokenize(cap)):
            chars[c] += 1

print(f"语料条数 {n} | 词级词表 {len(words)} | 字符级词表 {len(chars)}")
print(f"_EVIDENCE_MAX_DF = {vr._EVIDENCE_MAX_DF}\n")

PROBES = ["个人信息", "比较数字大小", "用户登录", "电影列表", "项目", "结构",
          "目录", "微信小程序", "tab1.wxml", "avatar.png"]

print(f"{'词':16s} {'词级DF':>9s} {'字符级(逐字DF)':>28s}   词级可通过门？")
print("-" * 88)
for p in PROBES:
    dw = words.get(p, 0) / max(n, 1)
    cds = ", ".join(f"{c}={chars.get(c, 0) / max(n, 1):.3f}" for c in p if len(p) <= 6)
    ok = "YES" if dw < vr._EVIDENCE_MAX_DF else "no"
    print(f"{p:16s} {dw:9.3f} {cds:>28s}   {ok}")

print()
print("对照：现有字符级 _content_tokens 对四张 ground-truth 问题抽出的证据词")
for q in ["项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？",
          "项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？",
          "项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？",
          "哪个微信小程序项目的目录里有 avatar.png 这个文件？"]:
    print(f"  Q: {q}")
    print(f"     _content_tokens(Q) = {sorted(vr._content_tokens(q))}")
    print(f"     jieba 词 (DF<{vr._EVIDENCE_MAX_DF}) = "
          f"{sorted(w for w in set(jieba.cut(q)) if len(w.strip()) >= 2 and words.get(w.strip(), 0) / max(n, 1) < vr._EVIDENCE_MAX_DF)}")
