#!/usr/bin/env python
"""AdaptiveTutor 端到端冒烟测试（2026-10-16）。

验证三件事，全部走真实 HTTP，不 mock：
  1. course-rag 服务可达（/health）
  2. course-rag /retrieve 真能返回带引用的文档
  3. tutor /api/explain/{course}/{kp} 能完成「FT → CR 检索 → FT 云端生成」并返回非空讲解

用法（默认按根 compose 的宿主端口）：
    python smoke/adaptive_tutor_smoke.py
    python smoke/adaptive_tutor_smoke.py --tutor-url http://127.0.0.1:8002 \
        --course-rag-url http://127.0.0.1:8010 --kp python.list.slice

退出码：0=全通过，1=任一环节失败。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(url, payload, timeout=180):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor-url", default="http://127.0.0.1:8001")
    ap.add_argument("--course-rag-url", default="http://127.0.0.1:8010")
    ap.add_argument("--course", default="python")
    ap.add_argument("--kp", default="python.list.slice")
    args = ap.parse_args()

    failures = []
    print("=" * 68)
    print("AdaptiveTutor end-to-end smoke")
    print("  tutor      =", args.tutor_url)
    print("  course-rag =", args.course_rag_url)
    print("=" * 68)

    # ---- 1. course-rag 可达 ----
    try:
        status, health = _get(args.course_rag_url + "/health", timeout=60)
        checks = health.get("checks") or []
        embed_ok = any(c.get("label") == "Ollama" and c.get("ok") for c in checks)
        print("[1] course-rag /health  -> HTTP %s, ok=%s, vectors=%s, ollama=%s"
              % (status, health.get("ok"), health.get("vector_store_count"), embed_ok))
        if not health.get("ok"):
            failures.append("course-rag /health 返回 ok=false")
    except Exception as e:
        print("[1] course-rag /health  -> FAIL %s" % e)
        failures.append("course-rag 不可达: %s" % e)

    # ---- 2. CR 检索真返回文档 ----
    try:
        t0 = time.time()
        _, body = _post(args.course_rag_url + "/retrieve",
                        {"question": "列表切片怎么用", "mode": "fast"}, timeout=180)
        docs = body.get("docs") or []
        print("[2] course-rag /retrieve -> %d docs in %.2fs" % (len(docs), time.time() - t0))
        for d in docs[:3]:
            print("      - %s (score=%s)" % (str(d.get("source"))[:60], d.get("score")))
        if not docs:
            failures.append("course-rag /retrieve 返回 0 文档")
    except Exception as e:
        print("[2] course-rag /retrieve -> FAIL %s" % e)
        failures.append("CR 检索失败: %s" % e)

    # ---- 3. FT 全链路：讲解入口（内部 FT→CR→云端生成）----
    try:
        t0 = time.time()
        url = "%s/api/explain/%s/%s" % (args.tutor_url, args.course, args.kp)
        status, body = _get(url, timeout=300)
        text = (body.get("explanation") or "").strip()
        print("[3] tutor /api/explain -> HTTP %s in %.2fs, explanation=%d chars"
              % (status, time.time() - t0, len(text)))
        print("      preview: %s" % text[:160].replace("\n", " "))
        if not text:
            failures.append("tutor 讲解为空（可能未走通 CR 或云端生成失败）")
    except urllib.error.HTTPError as e:
        print("[3] tutor /api/explain -> HTTP %s %s" % (e.code, e.read().decode("utf-8", "replace")[:200]))
        failures.append("tutor 讲解 HTTP %s" % e.code)
    except Exception as e:
        print("[3] tutor /api/explain -> FAIL %s" % e)
        failures.append("tutor 讲解失败: %s" % e)

    print("-" * 68)
    if failures:
        print("RESULT: FAIL (%d)" % len(failures))
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: PASS —— course-rag 可达 + 检索有文档 + tutor 全链路非空讲解")
    return 0


if __name__ == "__main__":
    sys.exit(main())