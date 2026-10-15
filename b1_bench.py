"""B1 视觉重排 A/B 基准：文本管线 vs 视觉重排，同一批查询、同一套 ground truth。

## 为什么这么设计

B1 的验收标准是「造一组**文本描述相似但图示不同**的题，看视觉重排能否把正确答案提到
rank=1，**必须有 before/after 对比数字**」。本脚本就是那个数字的产生器。

ground truth **不来自 caption**（那会循环论证——描述器写的字不能当标准），而是
**人工看图**确认的：`微信小程序开发实战` 里四张项目结构截图，caption 近乎同义，
但像素里各有一个独有的项目名 / 文件名：

| 文件 | 真实项目名 | 独有特征 | caption 里的向量名次 |
|---|---|---|---|
| `p045_i00.png` | 个人信息 | 有 `images/avatar.png` | 24 / 215 |
| `p082_i00.png` | 比较数字大小 | 无 images 文件夹 | 99 / 215 |
| `p167_i00.png` | 用户登录 | 无 images 文件夹 | 105 / 215 |
| `p267_i00.png` | 电影列表 | 有 `tab1/2/3.wxml` | 6 / 215 |

判别性 token（`个人信息` / `比较数字大小` / `avatar.png` / `tab1.wxml`）在**全部 caption
里出现 0 次** → 文本管线原理上无法区分这四张图。

## 三个 arm

- **A（before）**：`VISUAL_RERANK_ENABLED=false` → 现网文本链路（向量+BM25+RRF+reranker）
- **B（after）** ：开启 → 并入专用图像通道 + VLM 裁决 + 置顶

另外单独记录一个**归因指标**：目标图有没有被送进 VLM（`reachable`）。
`reachable=False` 而 `rank=None` ⇒ 失败发生在**召回**，视觉重排原理上无能为力；
`reachable=True` 而 `rank≠1` ⇒ 失败发生在**视觉判断**。两者修法完全不同。

用法（cwd 必须是 CourseRAG 根目录）：
    python b1_bench.py
"""
import csv
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _load_vlm_key():
    """未设置环境变量时，从 DSH 凭据读取 VLM key，避免把密钥写进仓库。"""
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
import visual_rerank  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "b1_bench_results.csv"

# (查询, 期望命中的图像块【全路径】, 该题的考点)
#
# 全部 ground truth 都由**人工看图**确认（不使用 caption，避免循环论证），并且每题
# 都验证过判别性 token 在 caption / 全量文本里的出现次数为 0 —— 即文本管线原理上
# 无法区分，只有看像素才能答对。
#
# ⚠️ 必须用**全路径**而不是裸文件名：实测 215 张 PNG 只有 159 个唯一 basename，
# 41 个 basename 跨书重名（`p045_i00.png` 同时存在于 Linux 第1章 / 第2章 / 第4章 /
# 微信小程序 4 本书里）。用 `"p045_i00.png" in image_path` 做子串匹配会把别书的
# 同名图判成命中 —— 这会让 before/after 数字凭空变好。
CASES = [
    # ---- 微信小程序开发实战：四张项目结构截图，caption 近乎同义 ----
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？",
     "data/images/微信小程序开发实战/p267_i00.png", "只存在于像素中的文件名"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？",
     "data/images/微信小程序开发实战/p045_i00.png", "只存在于像素中的文件名"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p267_i00.png", "只存在于像素中的项目名"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p045_i00.png", "只存在于像素中的项目名"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p167_i00.png", "只存在于像素中的项目名"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p082_i00.png", "只存在于像素中的项目名"),
]


def _norm(p):
    """路径归一化：反斜杠转正斜杠、去空白，便于跨平台精确比对。"""
    return re.sub(r"\s+", "", str(p or "").replace("\\", "/"))


def rank_of(docs, target):
    """目标图块在结果中的名次（1-based）；没进结果返回 None。

    用**全路径精确匹配**（而不是裸文件名子串）：41 个 basename 跨书重名，
    子串匹配会把别书的同名图算成命中。
    """
    want = _norm(target)
    for i, d in enumerate(docs, 1):
        if _norm(d.metadata.get("image_path")) == want:
            return i
    return None


def reachable(question, docs, sources, target):
    """目标图有没有被送进 VLM —— 即是否出现在「文本候选 ∪ 图像通道」的合并集里。"""
    merged = visual_rerank._merge_candidates(question, docs, sources)
    imgs = visual_rerank._collect_image_docs(merged)[: config.VISUAL_RERANK_MAX_IMAGES]
    # _collect_image_docs 给的是解析后的绝对路径，所以目标也要先解析成绝对路径再比
    want = _norm(visual_rerank._resolve_image_path(target))
    return any(_norm(a) == want for _, _, a in imgs)


def describe(docs):
    """把结果压成一行便于肉眼核对：图片块显示文件名，文本块显示 txt。"""
    parts = []
    for d in docs:
        if d.metadata.get("type") == "image":
            parts.append(pathlib.Path(d.metadata.get("image_path", "?")).name)
        else:
            parts.append("txt")
    return " ".join(parts)


def main():
    print(f"config: RERANKER={config.RERANKER_ENABLED} HYBRID={config.HYBRID_ENABLED} "
          f"model={config.VISUAL_RERANK_MODEL} mode={config.VISUAL_RERANK_MODE} "
          f"recall_k={config.VISUAL_RECALL_K} max_img={config.VISUAL_RERANK_MAX_IMAGES} "
          f"key={'set' if config.VISUAL_RERANK_API_KEY else 'MISSING'}")
    print()

    rows = []
    for q, target, note in CASES:
        # ---- arm A：文本链路（before）----
        config.VISUAL_RERANK_ENABLED = False
        t0 = time.time()
        docs_a = rag_core.retrieve_and_rerank(q)
        dt_a = time.time() - t0
        rank_a = rank_of(docs_a, target)

        # ---- arm B：+ 图像通道 + 视觉重排（after）----
        config.VISUAL_RERANK_ENABLED = True
        sources = rag_core._resolve_source_filter(q, None)
        # 用 arm A 的文本候选来算「可达性」：判断目标图能否被送进 VLM
        reach = reachable(q, docs_a, sources, target)
        t1 = time.time()
        docs_b = rag_core.retrieve_and_rerank(q)
        dt_b = time.time() - t1
        rank_b = rank_of(docs_b, target)

        print("=" * 100)
        print(f"Q: {q}")
        print(f"   考点={note}  期望={target}  可达={reach}")
        print(f"   A(before) rank={rank_a}  {dt_a:5.1f}s  -> {describe(docs_a)}")
        print(f"   B(after)  rank={rank_b}  {dt_b:5.1f}s  -> {describe(docs_b)}")
        rows.append({
            "question": q, "note": note, "expected": target,
            "reachable": reach,
            "rank_before": rank_a, "rank_after": rank_b,
            "dt_before_s": round(dt_a, 2), "dt_after_s": round(dt_b, 2),
        })

    config.VISUAL_RERANK_ENABLED = False

    # ---- 汇总 ----
    n = len(rows)
    b_top1 = sum(r["rank_before"] == 1 for r in rows)
    a_top1 = sum(r["rank_after"] == 1 for r in rows)
    b_top8 = sum(r["rank_before"] is not None for r in rows)
    a_top8 = sum(r["rank_after"] is not None for r in rows)
    reach_n = sum(r["reachable"] for r in rows)

    print()
    print("=" * 100)
    print(f"汇总（n={n} 题）")
    print(f"  目标图进入 VLM 视野（可达）      {reach_n}/{n}")
    print(f"  rank=1 准确率                    before {b_top1}/{n}   after {a_top1}/{n}")
    print(f"  进入结果（rank 非空）             before {b_top8}/{n}   after {a_top8}/{n}")
    avg_a = sum(r["dt_before_s"] for r in rows) / n
    avg_b = sum(r["dt_after_s"] for r in rows) / n
    print(f"  平均延迟                         before {avg_a:.1f}s   after {avg_b:.1f}s  (+{avg_b - avg_a:.1f}s)")
    print()
    print("  归因：可达=False 的题，失败在【召回】——视觉重排原理上无能为力；")
    print("        可达=True 而 rank≠1 的题，失败在【视觉判断】。")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n明细已写入 {OUT}")


if __name__ == "__main__":
    sys.exit(main())
