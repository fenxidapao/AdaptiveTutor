# -*- coding: utf-8 -*-
"""B3 探针：在 4 张 ground-truth 图上试 3 种 prompt，看哪种能把「判别性文字」写进 caption。

## 为什么先做这个而不是直接重跑 215 张

重跑 215 张图会改变整个语料 → 必须重跑全套评测（faithfulness / held-out / bench），
成本高且不可逆。而 B3 有两个可独立调整的杠杆：

  杠杆 1：**prompt**（现在的 prompt 要「2~3 句中文描述」→ 结构上就不会写出文件名）
  杠杆 2：**模型**（qwen2.5vl:7b → 更强的 VLM）

先花 2 分钟把杠杆 1 试出来，可能根本不需要动模型。

## 判据（客观、可复算）

4 张 ground-truth 图上各有一个**只存在于像素里**的判别性 token：

    p267_i00.png -> tab1.wxml / tab2.wxml / tab3.wxml   （电影列表）
    p045_i00.png -> avatar.png                           （个人信息）
    p167_i00.png -> 用户登录                              （根节点标签）
    p082_i00.png -> 比较数字大小                          （根节点标签）

这些 token 在现有 159 条 caption 里出现 **0 次** —— 所以任何一条能写出它的 prompt，
都是对文本检索面的实质扩充。

用法（cwd 必须是 CourseRAG 根目录）：
    python _b3_probe_prompt.py
"""
import base64
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parent

# ---- 4 张 ground-truth 图 + 各自的判别性 token（人工看图确认） ----
CASES = [
    ("data/images/微信小程序开发实战/p267_i00.png", ["tab1.wxml", "tab2.wxml", "tab3.wxml"]),
    ("data/images/微信小程序开发实战/p045_i00.png", ["avatar.png"]),
    ("data/images/微信小程序开发实战/p167_i00.png", ["用户登录"]),
    ("data/images/微信小程序开发实战/p082_i00.png", ["比较数字大小"]),
]

# ---- 三种 prompt ----
P_CURRENT = (  # 现有脚本 _caption_images.py:26-31 的原文（基线）
    "请用2~3句中文描述这张教材插图，要求覆盖："
    "1)图类型（操作步骤截图/软件界面截图/结构框图/流程图/原理图等）；"
    "2)图中的关键内容（界面元素、操作步骤、节点与关系等）；"
    "3)图中出现的数字、命令或文字内容。直接输出描述，不要开场白。"
)

P_TRANSCRIBE = (  # 纯抄录
    "请逐字抄出这张图中出现的**所有可见文字**，包括标题、根节点名称、文件夹名、"
    "文件名、按钮文字、代码文本。只输出抄到的文字，用逗号分隔，不要解释。"
)

P_HYBRID = (  # 描述 + 抄录（推荐候选：既保留语义检索面，又补上判别性文字）
    "请用 1~2 句中文描述这张教材插图的内容类型和主要元素，然后另起一行，"
    "以「图中文字：」开头，**逐字抄出图中出现的所有可见文字**"
    "（包括标题、根节点名称、文件夹名、文件名、按钮文字、命令、数字）。"
    "不要遗漏任何文件名，不要改写，不要翻译。"
)


def _key():
    k = os.environ.get("VISUAL_RERANK_API_KEY")
    if k:
        return k
    cred = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
    if cred.is_file():
        m = re.search(r"KUAIPAO_API_KEY:\s*(\S+)", cred.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise SystemExit("找不到 API key（设 VISUAL_RERANK_API_KEY 或写进 .credentials.yaml）")


API_BASE = os.environ.get("B3_API_BASE", "https://kuaipao.pro/v1")
MODEL = os.environ.get("B3_MODEL", "qwen3-vl-plus")


def call(png: pathlib.Path, prompt: str, key: str, detail="high", max_tokens=700):
    b64 = base64.b64encode(png.read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}",
                                                    "detail": detail}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            txt = data["choices"][0]["message"]["content"] or ""
            if isinstance(txt, list):
                txt = " ".join(str(c.get("text", "")) for c in txt if isinstance(c, dict))
            return txt.strip(), time.time() - t0, data.get("usage", {})
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    return f"[FAILED] {last}", time.time() - t0, {}


def main():
    key = _key()
    variants = [("CURRENT(基线)", P_CURRENT), ("TRANSCRIBE", P_TRANSCRIBE), ("HYBRID", P_HYBRID)]
    print(f"model={MODEL}  base={API_BASE}  detail=high  n_img={len(CASES)}\n")

    score = {name: 0 for name, _ in variants}
    total = 0
    for rel, tokens in CASES:
        png = ROOT / rel
        if not png.is_file():
            print(f"!! 缺文件 {rel}")
            continue
        print("=" * 100)
        print(f"图: {rel}")
        print(f"判别性 token: {tokens}")
        for name, prompt in variants:
            txt, dt, usage = call(png, prompt, key)
            flat = re.sub(r"\s+", "", txt)
            found = [t for t in tokens if re.sub(r"\s+", "", t) in flat]
            total += 1
            if found:
                score[name] += 1
            flag = "HIT " if found else "MISS"
            print(f"  [{flag}] {name:16s} ({dt:5.1f}s) 抄到 {found}  usage={usage.get('completion_tokens')}")
            print(f"         {txt[:220]}")
        print()

    print("=" * 100)
    print("命中汇总（判别性 token 被写进 caption 的图数 / 总图数）")
    for name, _ in variants:
        print(f"  {name:16s} {score[name]}/{len(CASES)}")
    print("\n判据：命中越多，文本检索面扩充越大；且 HYBRID 应同时保留描述句（不牺牲语义检索）。")


if __name__ == "__main__":
    main()
