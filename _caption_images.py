# -*- coding: utf-8 -*-
"""P2 步2：批量 caption（qwen2.5vl:7b @ Ollama HTTP API）。

- 输入：data/images/_manifest.csv（_extract_images.py 产物）
- 输出：data/images/_image_captions.csv（png_path, caption, elapsed_s）
- 断点续跑：已出现在输出 CSV 的 png_path 跳过；逐行 append 落盘
  （后台任务被回收不丢进度；查进度 wc -l _image_captions.csv）。
- 错峰纪律：批跑期间不要同时跑评估/reranker GPU 任务（P0 报告 §3）。
"""
import base64
import csv
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "data", "images", "_manifest.csv")
OUTCSV = os.path.join(ROOT, "data", "images", "_image_captions.csv")
API = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5vl:7b"

PROMPT = (
    "请用2~3句中文描述这张教材插图，要求覆盖："
    "1)图类型（操作步骤截图/软件界面截图/结构框图/流程图/原理图等）；"
    "2)图中的关键内容（界面元素、操作步骤、节点与关系等）；"
    "3)图中出现的数字、命令或文字内容。直接输出描述，不要开场白。"
)


def load_done():
    done = set()
    if os.path.exists(OUTCSV):
        with open(OUTCSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add(r["png_path"])
    return done


def caption_one(png_path):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 220},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return out.get("message", {}).get("content", "").strip(), time.time() - t0
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    return f"[CAPTION_FAILED] {last}", time.time() - t0


def main():
    with open(MANIFEST, encoding="utf-8-sig") as f:
        items = [r["png_path"] for r in csv.DictReader(f)]
    done = load_done()
    todo = [p for p in items if p not in done]
    print(f"total={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    new_file = not os.path.exists(OUTCSV)
    t_start = time.time()
    n = 0
    for i, rel in enumerate(todo):
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        text, dt = caption_one(path)
        with open(OUTCSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["png_path", "caption", "elapsed_s"])
                new_file = False
            w.writerow([rel, text, f"{dt:.1f}"])
        n += 1
        if n % 10 == 0 or n == len(todo):
            avg = (time.time() - t_start) / n
            remain = avg * (len(todo) - n)
            print(f"[{n}/{len(todo)}] avg={avg:.1f}s remain~{remain/60:.1f}min  last={rel}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
