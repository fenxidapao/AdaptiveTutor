"""B1 天花板探针：目标图在【专用图像通道】里的名次，以及判别性 token 能不能被看图读出。

## 为什么需要这个探针

`b1_probe_caption.py` 已证明判别性 token（`tab1.wxml` / `avatar.png` / `个人信息` …）
在**全部 159 条 caption 里出现 0 次**。而专用图像通道是按 **caption 的向量**排序的
——于是它继承了「描述器看不见像素」这个缺陷，目标图名次很深。

本探针把这件事量化成两个读数：
  1. 每道题的期望图在图像通道里的名次（决定 `VISUAL_RECALL_K` 要开多大才够）
  2. 直接问 VLM「请逐字抄出图上的文件名」→ 若判别性 token 能被抄出来，
     说明**像素里确实有这些字**，缺的只是索引层（= B3 的靶子），B1 的召回
     天花板由描述器质量决定，而不是由视觉重排本身决定。

用法（cwd 必须是 CourseRAG 根目录）：
    python b1_probe_ceiling.py
"""
import json
import os
import pathlib
import re
import sys
import time

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
import visual_rerank  # noqa: E402

CASES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？",
     "data/images/微信小程序开发实战/p267_i00.png"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？",
     "data/images/微信小程序开发实战/p045_i00.png"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p267_i00.png"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p045_i00.png"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p167_i00.png"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？",
     "data/images/微信小程序开发实战/p082_i00.png"),
]

# 每题期望图上真正该被读出来的判别性文字（人工看图确认）。
# ⚠️ 路径必须**带书名**：41 个 basename 跨书重名（`p045_i00.png` 在 4 本书里都有），
# rglob 裸文件名会随机命中别书的图 —— 实测曾把 Linux 安装向导截图当成微信小程序项目树。
EXPECT_TOKENS = {
    "data/images/微信小程序开发实战/p267_i00.png": ["tab1.wxml", "tab2.wxml", "tab3.wxml", "电影列表"],
    "data/images/微信小程序开发实战/p045_i00.png": ["avatar.png", "个人信息"],
    "data/images/微信小程序开发实战/p167_i00.png": ["用户登录"],
    "data/images/微信小程序开发实战/p082_i00.png": ["比较数字大小"],
}


def channel_ranks(question, sources):
    """在专用图像通道里找到每张图的名次（1-based），键为**归一化相对路径**。

    注意：`_image_channel` 的返回条数受 `config.VISUAL_RECALL_K` 限制，量真实名次
    必须先把 K 开大（见 main 里对 config 的临时改写），否则量到的只是「K=8 时在不在前 8」。
    """
    docs = visual_rerank._image_channel(question, sources)
    ranks = {}
    for i, d in enumerate(docs, 1):
        key = re.sub(r"\s+", "", str(d.metadata.get("image_path", "")).replace("\\", "/"))
        ranks.setdefault(key, i)
    return ranks


def main():
    # 量真实名次：把图像通道开到全库规模，否则名次会被 K 截断成「不在前 K」
    config.VISUAL_RECALL_K = 400
    print(f"config: recall_k={config.VISUAL_RECALL_K}（临时开大以量真实名次） "
          f"max_img={config.VISUAL_RERANK_MAX_IMAGES} mode={config.VISUAL_RERANK_MODE}\n")
    print("=" * 96)
    print("A. 期望图在【专用图像通道】里的名次（决定 VISUAL_RECALL_K 要多大才够）")
    print("=" * 96)

    worst = 0
    for q, target in CASES:
        sources = rag_core._resolve_source_filter(q, None)
        ranks = channel_ranks(q, sources)
        # 只在同书范围内找（图像通道被 source 过滤限定，名次是过滤后的）
        r = ranks.get(re.sub(r"\s+", "", target))
        worst = max(worst, r or 10 ** 6)
        print(f"  {target.split('/')[-1]:16s} 名次={str(r):>5s}  候选图共 {len(ranks)} 张   q={q[:30]}")
    print(f"\n  → 要 6/6 可达，VISUAL_RECALL_K 至少要开到 {worst if worst < 10**6 else 'N/A'}")
    print("     （名次按 caption 向量排序，所以这个数字就是『描述器盲区』的直接度量）")

    print()
    print("=" * 96)
    print("B. 直接问 VLM「抄出图上的文件名」—— 验证判别性 token 是否真的存在于像素里")
    print("=" * 96)

    for rel, want in EXPECT_TOKENS.items():
        path = pathlib.Path(rel)
        if not path.is_file():
            print(f"  {rel}: 文件不存在")
            continue
        name = path.name
        data_uri = visual_rerank._data_uri(path)
        payload = {
            "model": config.VISUAL_RERANK_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请逐字抄出这张截图里出现的所有文件名、文件夹名和项目名称。"
                                             "只输出抄到的文字，用逗号分隔，不要解释。"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "max_tokens": 400,
            "temperature": 0,
        }
        # 本机实测偶发 getaddrinfo failed（瞬时 DNS），必须重试——别把偶发当断网
        ok, data, dt = False, None, 0.0
        for attempt in range(3):
            t0 = time.time()
            ok, data = visual_rerank._post(payload, 60)
            dt = time.time() - t0
            if ok:
                break
            print(f"  {name}: 第 {attempt + 1} 次失败（{data}），重试…")
            time.sleep(2)
        if not ok:
            print(f"  {name}: 调用失败 {data}")
            continue
        text = data["choices"][0]["message"]["content"] or ""
        if isinstance(text, list):
            text = " ".join(str(c.get("text", "")) for c in text if isinstance(c, dict))
        flat = re.sub(r"\s+", "", text)
        found = [w for w in want if re.sub(r"\s+", "", w) in flat]
        flag = "OK " if found else "MISS"
        print(f"  [{flag}] {name} ({dt:4.1f}s)  抄到期望 token {found}/{want}")
        print(f"         原文: {text[:150]}")

    print("\n结论口径：若 B 段能抄出判别性 token（OK），说明像素里本来就有这些字，")
    print("          B1 召回不达标的原因是**索引层（caption）没把它们写进去** → 靶子是 B3；")
    print("          若 B 段也抄不出（MISS），说明这些字在图上不可读，B1 需换问法/换模型。")


if __name__ == "__main__":
    main()
