"""探针 3：目标图在**全部 215 张图**里的名次 + 它们的 caption 原文。

探针 2 读数：目标图在图像-only 通道 top-50 里 4/6 缺席，且把池宽从 16 拉到 200 也救不回来
→ 瓶颈不在「池宽」也不在「排序」，而在 caption 本身。

本探针把 k 拉到全库，看目标图到底排第几；并打印目标图 caption 与竞争图 caption 的原文，
定位「为什么文本召回想不起来」。

用法（cwd = CourseRAG 根）：
    python b1_probe_caption.py
"""
import csv
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _load_vlm_key():
    if os.environ.get("VISUAL_RERANK_API_KEY"):
        return
    cred = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
    if cred.is_file():
        m = re.search(r"KUAIPAO_API_KEY:\s*(\S+)", cred.read_text(encoding="utf-8"))
        if m:
            os.environ["VISUAL_RERANK_API_KEY"] = m.group(1)


_load_vlm_key()

import config  # noqa: E402
import rag_core  # noqa: E402

TARGETS = ["p267_i00.png", "p045_i00.png", "p167_i00.png", "p082_i00.png"]

# 1) 全库图像总数 + 目标图 caption 原文
print("=" * 100)
print("1. 目标图 caption 原文（索引里真正参与检索的文字）")
print("=" * 100)
cap_csv = pathlib.Path(config.DATA_DIR) / "images" / "_image_captions.csv"
caps = {}
with open(cap_csv, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        caps[pathlib.Path(r["png_path"]).name] = r["caption"]
for t in TARGETS:
    print(f"  {t}: {caps.get(t, '<无 caption>')}")

# 2) 全部图像块（直接取全库 image 类型）
all_imgs = rag_core.retrieve_vector("目录结构", k=1000, where={"type": "image"})
print()
print(f"2. 索引内图像块总数 = {len(all_imgs)}")

# 3) 每道题：目标图在全部图像里的名次
QUERIES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？", "p267_i00.png"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？", "p045_i00.png"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？", "p267_i00.png"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？", "p045_i00.png"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？", "p167_i00.png"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？", "p082_i00.png"),
]
print()
print("=" * 100)
print("3. 目标图在**全部图像**中的向量名次（k=1000 = 全库）")
print("=" * 100)
for q, target in QUERIES:
    docs = rag_core.retrieve_vector(q, k=1000, where={"type": "image"})
    rank = None
    for i, d in enumerate(docs, 1):
        if target in (d.metadata.get("image_path") or ""):
            rank = i
            break
    total = len(docs)
    pct = f"{rank / total:.1%}" if rank else "—"
    print(f"  {target:14s} rank={str(rank):>5s} / {total}  (百分位 {pct})   Q={q[:34]}…")

# 4) 关键词命中统计：判别性 token 在 215 条 caption 里出现几次
print()
print("=" * 100)
print("4. 判别性 token 在全部 caption 中的出现次数（0 = 文本管线原理上不可能召回）")
print("=" * 100)
for tok in ["tab1", "tab2", "tab3", "avatar.png", "电影列表", "个人信息", "用户登录", "比较数字大小"]:
    n = sum(1 for c in caps.values() if tok in c)
    print(f"  {tok:16s} 出现在 {n:3d} / {len(caps)} 条 caption 中")
