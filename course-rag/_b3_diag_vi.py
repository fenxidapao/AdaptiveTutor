"""B3 回归归因：换 caption 后那一道 recall 掉到 0 的题，检索结果到底变了什么。

冻结 24 题里只有 2 题 recall 下降（`vi 中 :%s/oldStr/newStr/gc…` 1.00→0.00、
`Streamlit st.file_uploader()…` 0.75→0.67），其余 22 题逐位不变。
本脚本对这两道题并排打印 v1/v3 两个库的 top-8 组成，把「机制」从推断变成实测。

跑法（cwd = CourseRAG 根）：
    # v3（换描述器后的生产库）
    python _b3_diag_vi.py > v3.txt
    # v1（换描述器之前的库，用备份索引 + 旧 caption）
    $env:PERSIST_DIRECTORY='./chroma_db_v1'
    $env:IMAGE_CAPTIONS_CSV='_image_captions_v1_qwen25vl.csv'
    python _b3_diag_vi.py > v1.txt
"""

import json
import sys

import config
import rag_core

QUESTIONS = [
    "vi 中 :%s/oldStr/newStr/gc 替换命令各部分的含义是什么？",
    "Streamlit 中文件上传组件 st.file_uploader() 如何限制文件类型？",
]


def dump(q: str) -> dict:
    docs = rag_core.retrieve_and_rerank(q)
    rows = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        rows.append({
            "rank": i + 1,
            "type": meta.get("type") or "text",
            "source": (meta.get("source") or "").replace("\\", "/").split("/")[-1],
            "page": meta.get("page"),
            "image_path": meta.get("image_path"),
            "head": (d.page_content or "")[:110].replace("\n", " "),
        })
    n_img = sum(1 for r in rows if r["type"] == "image")
    return {"question": q, "n": len(rows), "n_image": n_img, "rows": rows}


def main() -> int:
    out = {
        "persist_dir": config.PERSIST_DIRECTORY,
        "captions_csv": getattr(config, "IMAGE_CAPTIONS_CSV", "<none>"),
        "visual_rerank": config.VISUAL_RERANK_ENABLED,
        "cases": [dump(q) for q in QUESTIONS],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
