"""B2 视觉作答 A/B 验收基准（真 HTTP 端到端）。

## 它回答什么问题

B2 的诉求是「让像素真的进模型」。本脚本给出**可复跑的数字**，而不是「已完成」。
三臂对照，同一道题、同一把尺子（答案关键词是否出现在回答里）：

  臂 1「B2 之前」  把该图在索引里的 **caption 文本**当上下文喂给模型
                   —— 这是 B2 之前 FT 唯一的通路（`rag.py` 只读 score/content/source）
  臂 2「B2 之后」  `vision.answer_with_images`：把**原图像素**作为 content 块喂给视觉模型
  臂 3「无上下文」 什么都不给（对照；证明模型不会凭空知道，避免把运气当能力）

## 两条例题（ground truth 由人工读图确定，不是从 caption 推导）

  A. `data/images/Linux第1章概述/p006_i00.png`
     像素：一位短发男性，双臂交叉，穿**浅灰色带领 POLO 衫**，背景是模糊的弧形钢结构。
     caption（生产 v3，59 字）：只写「人物肖像照片…双臂交叉、面带微笑的中年男性…」，
     **完全没有衣着信息**。问「上衣是什么颜色」→ 文本臂没有依据，像素臂有。
     （旧 v1 caption 曾把这张照片写成「软件界面截图」——即旧描述器连图类型都判错。）

  B. `data/images/Linux第1章概述/p008_i00.png`
     像素：**黑白线条画的牛科动物巨头**（弯曲大角、拟人化表情）。
     v1 旧 caption 写的是「带有**羊角**的**人物头像**，可能代表某个软件或系统的图标」
     → 文本臂会照着 caption 答错；像素臂应当答对。
     这条特意用「caption 会骗人」的例子：B2 的价值正是**不再依赖 caption 的质量**。

脚本会自己复算并断言「答案关键词不在所用 caption 里」，所以命题不靠我口头保证。

## 契约臂（E1，不发 LLM）

先证明「像素拿得到」：FT → CourseRAG `/retrieve` → 断言返回里出现
`type == "image"` 与 `image_path`，经 `rag.image_doc_paths` 取出来，
再经 `vision.fetch_image` 走 `/images/{rel_path}` 取到真实字节数。
B1/B3 之前 `rag.py` 把这两个字段丢掉了（`:60-66` 只读 score/content/source），
所以这一臂正是 B2 新增的能力。

## 用法

    # 先起 CourseRAG（本机实测端口 8010；8000 可能被别的东西占着）
    #   cd <CourseRAG>; $env:SKIP_REBUILD='1'
    #   python -m uvicorn main:app --host 127.0.0.1 --port 8010
    cd <FeynmanTutor>
    $env:RAG_BASE_URL='http://127.0.0.1:8010'
    $env:VISION_ENABLED='1'
    $env:VISION_MODEL='qwen3-vl-plus'      # 实测 0.96s/图，比默认模型快且稳
    python b2_bench.py

输出：控制台 + `b2_bench_results.csv`
"""

import csv
import json
import os
import sys
import time
import urllib.request

import config
import model
import rag
import vision

# ---------------------------------------------------------------------------
# 预热：CourseRAG 的 accurate 模式要现载 bge-reranker（首次 15~41s），
# 而 FT 的 rag.py 对 /retrieve 固定 15s 超时 → 冷启动时第一条问句必然超时，
# 会把「服务器没热」误报成「B2 不通」。这里先直接打一次 /retrieve 把模型热起来。
# ---------------------------------------------------------------------------
def warmup(timeout: float = 180.0) -> None:
    url = f"{config.RAG_BASE_URL}/retrieve"
    body = json.dumps({"question": "预热", "mode": config.RAG_MODE}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        print(f"[warmup] {url} ok  {time.time() - t0:.1f}s（reranker 已载入）")
    except Exception as e:
        print(f"[warmup] {url} 失败 {time.time() - t0:.1f}s：{type(e).__name__}: {e}")
        print("[warmup] 继续跑，但 E1 可能整体超时——先确认 CourseRAG 起来了")

# ---------------------------------------------------------------------------
# E1 契约臂：这些问句在 CourseRAG 上都实测能召回图像块
# 注：第 3 条在换描述器（B3）之后 top-8 里不再含图像块（`微信小程序开发实战/p020`
#     的官方图片命中率由 rank=1 变为 MISS，属 B3 的已知回归，见 B3 报告）。
#     留着它是为了让契约臂**如实报告**，而不是挑一条好看的。
# ---------------------------------------------------------------------------
CONTRACT_QUESTIONS = [
    "Linux 文件系统树中 /bin、/etc、/home 各存放什么？",
    "Streamlit 中文件上传组件 st.file_uploader() 如何限制文件类型？",
    "创建微信小程序项目时需要填写或选择哪些设置项？",
]

# ---------------------------------------------------------------------------
# E2 消融臂：例题
# ---------------------------------------------------------------------------
# caption_text 是**逐字抄自** CourseRAG 的 `data/images/_image_captions.csv`；
# v1_caption_text 抄自 `_image_captions_v1_qwen25vl.csv`（换描述器之前的留档）。
# 抄成常量而不是跨仓读文件，是为了让本脚本不依赖宿主机上 CR 的绝对路径。
CASES = [
    {
        "name": "A-衣着（caption 对此沉默）",
        "question": "这张插图里的男性穿的上衣是什么颜色的？是带领子的吗？",
        "image_path": "data/images/Linux第1章概述/p006_i00.png",
        "expect": ["灰"],
        "expect_note": "浅灰色带领 POLO 衫（人工读图像素确认）",
        "caption_source": "data/images/_image_captions.csv",
        "caption_text": (
            "这是一张人物肖像照片，主要元素为一位双臂交叉、面带微笑的中年男性，"
            "背景为模糊的弧形建筑结构。\n\n图中文字：无可辨认文字"
        ),
        "forbid_in_caption": ["灰", "上衣", "衬衫", "POLO", "领子", "颜色"],
    },
    {
        "name": "B-动物（caption 说的是错的）",
        "question": "这张插图里画的是什么动物？请只回答动物名称。",
        "image_path": "data/images/Linux第1章概述/p008_i00.png",
        "expect": ["牛"],
        "expect_note": "牛科动物（弯曲大角的牛头线条画），人工读图像素确认",
        "caption_source": "data/images/_image_captions_v1_qwen25vl.csv",
        "caption_text": "这张插图属于结构框图类型，图中展示了一个带有羊角的人物头像，可能代表某个软件或系统的图标。",
        "forbid_in_caption": ["牛"],
    },
]

SYSTEM = "你是课程助教。只根据给定材料回答；材料里没有的信息就明确说不知道，不要猜。"


def _hit(answer: str, keywords: list[str]) -> bool:
    return any(k in (answer or "") for k in keywords)


def _short(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def contract_arm() -> list[dict]:
    """E1：证明「FT 能拿到 CourseRAG 的原图」，不发 LLM。"""
    print("=" * 78)
    print("E1 契约臂：FT → CourseRAG /retrieve → image_path → /images/{path} 取字节")
    print(f"   RAG_BASE_URL = {config.RAG_BASE_URL}")
    print(f"   RAG_MODE     = {config.RAG_MODE}")
    print("=" * 78)
    rows = []
    for q in CONTRACT_QUESTIONS:
        rec = {"question": q, "n_docs": 0, "n_image_docs": 0, "image_paths": [],
               "fetched_bytes": [], "ok": False}
        try:
            docs = rag._retrieve_raw(q)
        except Exception as e:
            print(f"[ERR] {q} → {type(e).__name__}: {e}")
            rec["error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
            continue
        rec["n_docs"] = len(docs)
        img_paths = rag.image_doc_paths(docs)
        rec["n_image_docs"] = sum(1 for d in docs if d.get("type") == "image")
        rec["image_paths"] = img_paths[:3]
        for p in img_paths[:2]:
            data = vision.fetch_image(p)
            rec["fetched_bytes"].append(len(data) if data else 0)
        rec["ok"] = bool(rec["n_image_docs"] and img_paths and all(rec["fetched_bytes"]))
        print(f"\nQ: {q}")
        print(f"   docs={rec['n_docs']}  其中图像块={rec['n_image_docs']}  "
              f"image_doc_paths={len(img_paths)}  取到字节={rec['fetched_bytes']}  ok={rec['ok']}")
        for i, d in enumerate(docs[:3]):
            print(f"   [{i}] type={d.get('type')!r} score={d.get('score')} "
                  f"image_path={d.get('image_path')!r} source={d.get('source')!r}")
        rows.append(rec)
    print(f"\nE1 小结：{sum(1 for r in rows if r['ok'])}/{len(rows)} 条问句成功取到原图字节")
    return rows


def ablation_arm() -> list[dict]:
    """E2：三臂消融，答案只在像素里的题。"""
    print("\n" + "=" * 78)
    print("E2 消融臂：文本(旧通路) vs 像素(B2) vs 无上下文(对照)")
    print(f"   VISION_ENABLED = {config.VISION_ENABLED}   VISION_MODEL = {config.VISION_MODEL}")
    print(f"   DEEPSEEK_MODEL = {config.DEEPSEEK_MODEL}   available() = {vision.available()}")
    print("=" * 78)
    rows = []
    for case in CASES:
        leaked = [w for w in case["forbid_in_caption"] if w in case["caption_text"]]
        assert not leaked, f"{case['name']}: 答案词泄漏进 caption {leaked}，选题不成立"
        print(f"\n### {case['name']}")
        print(f"    图: {case['image_path']}")
        print(f"    caption 来源: {case['caption_source']}（已断言不含 {case['forbid_in_caption']}）")
        print(f"    问: {case['question']}")
        rec = {"case": case["name"], "question": case["question"],
               "image_path": case["image_path"], "expect": ",".join(case["expect"])}

        arms = []
        # 臂 1：文本（B2 之前的通路）
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"材料：\n{case['caption_text']}\n\n问题：{case['question']}"}]
        arms.append(("text_caption", lambda: model.chat(msgs, max_tokens=400, caller="b2_text")))
        # 臂 2：像素（B2）
        arms.append(("vision_pixels", lambda: vision.answer_with_images(
            case["question"], [case["image_path"]],
            system=SYSTEM + " 请先仔细观察图片。", max_tokens=400, caller="b2_vision")))
        # 臂 3：无上下文（对照）
        msgs0 = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"问题：{case['question']}"}]
        arms.append(("no_context", lambda: model.chat(msgs0, max_tokens=400, caller="b2_nocontext")))

        for arm_name, fn in arms:
            t0 = time.time()
            try:
                ans = fn()
                err = ""
            except Exception as e:
                ans, err = "", f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            hit = _hit(ans, case["expect"])
            rec[f"{arm_name}_hit"] = hit
            rec[f"{arm_name}_sec"] = round(dt, 1)
            rec[f"{arm_name}_answer"] = ans
            if err:
                rec[f"{arm_name}_error"] = err
            print(f"    [{arm_name:14s}] hit={str(hit):5s} {dt:5.1f}s  "
                  f"{'ERR ' + err if err else _short(ans, 160)}")
        rows.append(rec)
    return rows


def main() -> int:
    print(f"课程助教 B2 视觉作答基准 | python={sys.version.split()[0]}")
    warmup()
    e1 = contract_arm()
    e2 = ablation_arm()

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    e1_ok = sum(1 for r in e1 if r["ok"])
    print(f"E1 契约：{e1_ok}/{len(e1)} 条问句端到端取到原图字节")
    for r in e2:
        t = r["text_caption_hit"]
        v = r["vision_pixels_hit"]
        n = r["no_context_hit"]
        print(f"E2 {r['case'][:22]:22s} 文本={t} 像素={v} 无上下文={n}  (期望含 {r['expect']})")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b2_bench_results.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "question_or_case", "detail_json"])
        for r in e1:
            w.writerow(["E1_contract", r["question"], json.dumps(r, ensure_ascii=False)])
        for r in e2:
            w.writerow(["E2_ablation", r["case"], json.dumps(r, ensure_ascii=False)])
    print(f"\n明细已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
