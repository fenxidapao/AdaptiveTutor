# -*- coding: utf-8 -*-
"""探针：证据门离线对照表 —— 用 4 张 ground-truth 图的**真实 caption** 当证据。

不需要调 VLM，秒级可复跑。判据很硬：

  对每道题，把「正确答案图的 caption」和「3 张干扰图的 caption」都当证据过门，
  理想结果 = **只有正确答案图通过**（4 题 × 4 图 = 16 格，命中 4 格 + 误放 0 格）。

这正是 B1 证据门存在的唯一理由：VLM 会把无关图也报成命中（`b1_bench` 早期
`promote` 模式实测 6 题里 5 题被硬塞无关图到 rank 1），门必须把那些挡掉，
同时**不能把真命中挡掉**。

历史教训（修这个门的原因）：字符级 token 让「个人信息」永远当不了证据词，
VLM 报对了也会被拒 → 假阴性。见 `_probe_df_jieba.py`。

用法（cwd 必须是 CourseRAG 根目录）：
    python _probe_gate.py
"""
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
import visual_rerank as vr

CAP = pathlib.Path(config.DATA_DIR) / "images" / config.IMAGE_CAPTIONS_CSV
captions = {}
with open(CAP, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        captions[row["png_path"].replace("/", "\\")] = row.get("caption", "")

BOOK = "data\\images\\微信小程序开发实战\\"

# (问题, 正确答案文件名, 该题的判别性词)
CASES = [
    ("哪个微信小程序项目的目录里同时有 tab1.wxml、tab2.wxml、tab3.wxml 这三个文件？", "p267_i00.png"),
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？", "p045_i00.png"),
    ("项目名叫「电影列表」的那个微信小程序，目录结构是怎样的？", "p267_i00.png"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？", "p045_i00.png"),
    ("项目名叫「比较数字大小」的那个微信小程序，目录结构是怎样的？", "p082_i00.png"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？", "p167_i00.png"),
]
DISTRACTORS = ["p045_i00.png", "p082_i00.png", "p167_i00.png", "p267_i00.png"]

print(f"caption 语料 = {CAP}（{len(captions)} 条）")
print(f"_EVIDENCE_MAX_DF = {vr._EVIDENCE_MAX_DF}  停用词 {len(vr._STOP)} 个")
print()
print("=== 一、问题侧抽出的判别性词（决定门能否工作） ===")
for q, want in CASES:
    print(f"  {want}  {sorted(vr._content_tokens(q))}")
print()

print("=== 二、离线对照矩阵（证据 = 该图真实 caption） ===")
hdr = "  " + "问题 \\ 证据图".ljust(46) + "".join(f"{d[:9]:>11s}" for d in DISTRACTORS)
print(hdr)
print("  " + "-" * (46 + 11 * len(DISTRACTORS)))

tp = fp = fn = tn = 0
for q, want in CASES:
    cells = []
    for d in DISTRACTORS:
        cap = captions.get(BOOK + d, "")
        ok = vr._evidence_supports(q, cap)
        if d == want and ok:
            cells.append("  ✓命中"); tp += 1
        elif d == want:
            cells.append("  ✗漏判"); fn += 1
        elif ok:
            cells.append("  ⚠误放"); fp += 1
        else:
            cells.append("   ·拒"); tn += 1
    short = q.replace("哪个微信小程序项目的目录里", "目录里")[:44]
    print("  " + short.ljust(46) + "".join(f"{c:>11s}" for c in cells))

print()
print(f"  真命中通过  {tp}/6    真命中被漏判  {fn}/6")
print(f"  干扰图误放  {fp}/18   干扰图正确拒掉  {tn}/18")
print()
print("  判读：`漏判` 必须为 0（否则 B1 对中文静默失效）；`误放` 越低越好。")

print()
print("=== 三、历史假阳性回归（必须全部拒掉） ===")
FALSE_POS = [
    ("哪个微信小程序项目的目录里有 avatar.png 这个文件？", "images, guest.png", "文件名门"),
    ("项目名叫「个人信息」的那个微信小程序，目录结构是怎样的？",
     "pages, index, index.js, index.json, index.wxml, index.wxss, .eslintrc.js, app.js", "通用目录名"),
    ("项目名叫「用户登录」的那个微信小程序，目录结构是怎样的？",
     "pages, index, index.js, index.json, index.wxml, index.wxss, .eslintrc.js, app.js", "通用目录名"),
]
for q, ev, why in FALSE_POS:
    got = vr._evidence_supports(q, ev)
    flag = "❌ 误放（回归！）" if got else "✅ 拒掉"
    print(f"  {flag}  [{why}] evidence={ev[:60]!r}")
