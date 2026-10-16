# -*- coding: utf-8 -*-
"""审计：B1 的 6 道「像素题」里，判别性 token 到底是不是真的**只存在于像素**。

## 为什么必须做这一步

B1 的 bench 数字（before 0/6 → after 2/6）建立在「这些 token 文本管线搜不到」的前提上。
`b1_probe_caption.py` 当时只查了 caption 语料 + `data/*.md`，**漏了 PDF 抽取出来的正文**
（`data/教材/*.pdf` 在索引时被抽成文本块）。刚才重建索引后抽验发现
`tab1.wxml` 在库里命中 **6 块**，其中 **5 块是文本块** —— 说明那个前提至少对某几道题不成立。

本脚本对**索引里的真实文本语料**（type != image）逐 token 核查：
  - 命中 0  → 该题是干净的「像素题」，B1 的归因成立
  - 命中 >0 → 该题文本可答，**必须从 B1 的像素题集合里剔除或改写**，否则 before/after 数字是虚的

用法（cwd 必须是 CourseRAG 根目录）：
    python _b1_audit_tokens.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
import rag_core

# 6 道题的判别性 token（b1_bench.py 的 CASES 里逐题标注的）
CASES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？",
     "data/images/微信小程序开发实战/p267_i00.png",
     ["tab1.wxml", "tab2.wxml", "tab3.wxml"]),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？",
     "data/images/微信小程序开发实战/p045_i00.png",
     ["avatar.png"]),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p267_i00.png",
     ["电影列表"]),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p045_i00.png",
     ["个人信息"]),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p167_i00.png",
     ["用户登录"]),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p082_i00.png",
     ["比较数字大小"]),
]


def main():
    col = rag_core.get_vector_store(auto_rebuild=False)
    print(f"IMAGE_CAPTIONS_CSV = {config.IMAGE_CAPTIONS_CSV}")
    print(f"库内总块数 = {col.count()}")
    n_text = len(col.get(where={"type": {"$ne": "image"}}, include=["metadatas"])["metadatas"])
    n_img = len(col.get(where={"type": "image"}, include=["metadatas"])["metadatas"])
    print(f"文本块 = {n_text} | 图像块 = {n_img}\n")

    verdict = []
    for q, target, toks in CASES:
        print("=" * 96)
        print(f"Q: {q}")
        print(f"   目标图: {target.rsplit('/', 1)[-1]}")
        clean = True
        for tok in toks:
            got = col.get(where_document={"$contains": tok}, include=["metadatas", "documents"])
            ms = [m or {} for m in got["metadatas"]]
            n_txt = sum(1 for m in ms if m.get("type") != "image")
            n_img_hit = sum(1 for m in ms if m.get("type") == "image")
            flag = "OK  " if n_txt == 0 else "LEAK"
            if n_txt:
                clean = False
            print(f"   [{flag}] {tok:14s} 文本块命中 {n_txt:2d} | 图像块命中 {n_img_hit}")
            for m, d in zip(ms, got["documents"]):
                if (m or {}).get("type") != "image":
                    src = str(m.get("source", "")).rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                    i = d.find(tok)
                    ctx = d[max(0, i - 45): i + 45].replace("\n", " ")
                    print(f"          ← {src}  …{ctx}…")
        verdict.append((clean, q, target))

    print("\n" + "=" * 96)
    print("结论：")
    n_clean = sum(1 for c, _, _ in verdict if c)
    for c, q, t in verdict:
        print(f"  [{'干净' if c else '文本可答'}] {t.rsplit('/', 1)[-1]:14s} {q[:52]}")
    print(f"\n干净像素题 {n_clean}/{len(verdict)}；文本可答 {len(verdict) - n_clean}/{len(verdict)}")


if __name__ == "__main__":
    main()
