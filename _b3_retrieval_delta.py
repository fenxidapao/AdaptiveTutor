# -*- coding: utf-8 -*-
"""B3 检索层面 delta：旧 caption 库 vs 新 caption 库，同一批问题，逐题对比 top-8。

用途：回答「B3 换描述器到底改变了什么」。
评测层的 context_recall 有裁判污染（见 B3_CAPTIONER.md §3），所以必须回到**检索层**量：

    PERSIST_DIRECTORY=./chroma_db_v3 IMAGE_CAPTIONS_CSV=_image_captions.csv           python _b3_retrieval_delta.py dump v3.json
    PERSIST_DIRECTORY=./chroma_db_v1 IMAGE_CAPTIONS_CSV=_image_captions_v1_qwen25vl.csv  python _b3_retrieval_delta.py dump v1.json
    python _b3_retrieval_delta.py compare v1.json v3.json

对比项：每题 top-8 的 (内容 md5, 类型, source, image_path, 重排分)。
输出必须是**稳定可比**的：不做任何 LLM 调用，所以完全确定性。
"""
import hashlib
import json
import os
import sys
import faulthandler
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("SKIP_REBUILD", "1")
os.environ.setdefault("VISUAL_RERANK_ENABLED", "false")

# 卡住时能看到栈：本机 8GB 显存要同时装 Ollama 常驻模型 + reranker，
# 实测会出现 reranker.predict 长时间占满 GPU 的情况（见 B3_CAPTIONER.md §5.3）。
faulthandler.dump_traceback_later(180, repeat=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def dump(out_path):
    import config
    import rag_core
    from eval_data_test import TEST_SET

    print(f"PERSIST_DIRECTORY  = {config.PERSIST_DIRECTORY}")
    print(f"IMAGE_CAPTIONS_CSV = {config.IMAGE_CAPTIONS_CSV}")
    print(f"VISUAL_RERANK_ENABLED = {config.VISUAL_RERANK_ENABLED}")
    rows = []
    for i, item in enumerate(TEST_SET, 1):
        q = item["question"]
        docs = rag_core.retrieve_and_rerank(q)
        top = []
        for d in docs:
            md = d.metadata or {}
            top.append({
                "md5": hashlib.md5(d.page_content.encode("utf-8")).hexdigest()[:12],
                "type": md.get("type") or "text",
                "source": os.path.basename(str(md.get("source") or "")),
                "image_path": md.get("image_path") or "",
                "score": round(float(md.get("rerank_score") or 0.0), 6),
                "len": len(d.page_content),
            })
        rows.append({"i": i, "question": q, "top": top})
        print(f"[{i}/24] {q[:40]}  n={len(top)}  img={sum(1 for t in top if t['type']=='image')}")
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写出 {out_path}（{len(rows)} 题）")


def compare(a_path, b_path):
    a = {r["i"]: r for r in json.loads(Path(a_path).read_text(encoding="utf-8"))}
    b = {r["i"]: r for r in json.loads(Path(b_path).read_text(encoding="utf-8"))}
    same_all = 0
    print(f"{'#':>3} {'题目':44s} {'集合相同':8s} {'顺序相同':8s} {'旧图数':5s} {'新图数':5s} 差异")
    for i in sorted(a):
        ta = [x["md5"] for x in a[i]["top"]]
        tb = [x["md5"] for x in b[i]["top"]]
        set_same = set(ta) == set(tb)
        order_same = ta == tb
        if order_same:
            same_all += 1
        ia = sum(1 for x in a[i]["top"] if x["type"] == "image")
        ib = sum(1 for x in b[i]["top"] if x["type"] == "image")
        notes = []
        if not set_same:
            notes.append(f"集合内 {len(set(ta) ^ set(tb))} 条不同")
        if set_same and not order_same:
            notes.append("仅顺序不同")
        if ia != ib:
            notes.append(f"图像块 {ia}→{ib}")
        print(f"{i:>3} {a[i]['question'][:42]:44s} {str(set_same):8s} {str(order_same):8s} "
              f"{ia:<5} {ib:<5} {'; '.join(notes)}")
    print(f"\n24 题里 top-8 完全一致（含顺序）= {same_all}/24")
    n_set = sum(1 for i in a if set(x['md5'] for x in a[i]['top']) == set(x['md5'] for x in b[i]['top']))
    print(f"24 题里 top-8 集合一致（不计顺序）= {n_set}/24")
    n_img = sum(1 for i in a if sum(1 for x in a[i]['top'] if x['type'] == 'image')
                != sum(1 for x in b[i]['top'] if x['type'] == 'image'))
    print(f"24 题里 top-8 图像块条数发生变化的题数 = {n_img}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dump":
        dump(sys.argv[2])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
