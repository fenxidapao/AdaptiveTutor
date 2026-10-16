"""B2 视觉作答：把 CourseRAG 里的**原图像素**取回来，作为 content 块喂给视觉模型。

## 为什么需要它（这条路以前是断的）

B1/B3 之前，图片在整条链路里只是一段**描述文字**：CourseRAG 把 PNG 交给本地描述器
（`qwen2.5vl:7b`）转成一句话，再把那句话当普通文本块嵌入；FeynmanTutor 从
`/retrieve` 只读 `score/content/source`（`rag.py:60-66`），连「这条其实是一张图」都
不知道。**像素从没进过模型。**

## 两个实测约束（2026-10-14/15，中转 kuaipao.pro/v1）

1. **图片必须放 user 消息**。放 system 时：`qwen3-vl-plus` 直接 HTTP 400
   （`An incorrect modal \`image\` was entered ... placed in the wrong position`）；
   `deepseek-v4.1-flash` 更危险——不报错，静默把 token 预算烧在推理上、返回空 content。
   所以本模块只把图块拼进 **user** 消息。
2. 视觉能力是真的，不是文本冒充：用自造的四象限色块图（纯文本模型必错）实测，
   `qwen3-vl-plus` 与 `deepseek-v4.1-flash` 都答对「左上=红 右上=绿 左下=蓝 右下=黄」。

## 取图路径

走 CourseRAG 新增的 `GET {RAG_BASE_URL}/images/{image_path}`，**不在本端拼宿主机
绝对路径**（交接文档明确警告过：写死宿主机路径换机器即失效）。因此 FT 与服务
不同容器/不同盘符都能用。

## 对外接口

    available() -> bool                          当前配置下是否可以走视觉
    fetch_image(image_path) -> bytes | None      取原图字节（失败返回 None 并打印原因）
    build_image_blocks(image_paths) -> list[dict] 拼成 OpenAI 兼容的 image_url 块
    answer_with_images(question, image_paths, ...) -> str  带图提问（图片在 user 消息）
"""

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

import config

# 后缀 → MIME（CourseRAG 的 /images 端点已限制白名单，这里只是给 data URI 用）
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def available() -> bool:
    """当前配置下是否可以走视觉作答。

    要求：① VISION_ENABLED 打开；② 配了 API key。
    （模型是否真能看图由调用方实测决定，这里不假装知道。）
    """
    return bool(config.VISION_ENABLED and config.DEEPSEEK_API_KEY)


def _image_url(image_path: str) -> str:
    """把索引里的 image_path 拼成 CourseRAG 的取图 URL。

    image_path 形如 `data/images/微信小程序开发实战/p267_i00.png`（反斜杠、中文都有）。
    只对路径段做百分号编码，保留 `/` —— 整段 quote 会把分隔符编成 %2F，路由就断了。
    """
    raw = (image_path or "").replace("\\", "/").lstrip("/")
    return f"{config.RAG_BASE_URL}/images/{urllib.parse.quote(raw, safe='/')}"


def fetch_image(image_path: str, timeout: float | None = None) -> bytes | None:
    """从 CourseRAG 取一张原图的字节。任何失败都返回 None 并打印原因（不抛）。"""
    if not image_path:
        return None
    url = _image_url(image_path)
    limit = config.VISION_MAX_IMAGE_BYTES
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(
            req, timeout=timeout or config.VISION_IMAGE_TIMEOUT
        ) as resp:
            data = resp.read(limit + 1)
    except urllib.error.HTTPError as e:
        print(f"[i] 取图失败（HTTP {e.code}）：{image_path}")
        return None
    except Exception as e:
        print(f"[i] 取图失败（{type(e).__name__}: {e}）：{image_path}")
        return None
    if len(data) > limit:
        print(f"[i] 取图跳过（超过 {limit} 字节上限）：{image_path}")
        return None
    if not data:
        print(f"[i] 取图跳过（0 字节）：{image_path}")
        return None
    return data


def build_image_blocks(image_paths: list[str], max_images: int | None = None) -> list[dict]:
    """取图并拼成 OpenAI 兼容的 content 块：[{"type":"image_url", ...}, ...]。

    取不到的图**跳过而不报错**——少一张图退化为纯文本，比整次讲解失败好。
    """
    cap = max_images if max_images is not None else config.VISION_MAX_IMAGES
    blocks: list[dict] = []
    for p in list(image_paths)[:cap]:
        data = fetch_image(p)
        if data is None:
            continue
        suffix = ("." + p.rsplit(".", 1)[-1].lower()) if "." in p else ".png"
        mime = _MIME.get(suffix, "image/png")
        b64 = base64.b64encode(data).decode("ascii")
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return blocks


def answer_with_images(question: str, image_paths: list[str],
                       system: str | None = None, temperature: float = 0.3,
                       max_tokens: int = 1200, caller: str = "vision") -> str:
    """带原图提问（图片在 **user** 消息里）。

    返回模型文本；一张图都没取到时抛 ValueError（调用方应回退纯文本链路，
    不要静默给出「没看图」的答案冒充看图答案）。
    """
    import model  # 延迟 import：避免与 model 的模块级循环依赖

    blocks = build_image_blocks(image_paths)
    if not blocks:
        raise ValueError("没有取到任何原图，无法进行视觉作答")
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": question}, *blocks],
    })
    return model.chat(messages, temperature=temperature, max_tokens=max_tokens,
                      caller=caller, model_name=config.VISION_MODEL)


def describe_for_debug(image_paths: list[str]) -> str:
    """只做「能不能取到图」的诊断（不发 LLM），给验收脚本用。"""
    out = []
    for p in image_paths:
        data = fetch_image(p)
        out.append({
            "image_path": p,
            "url": _image_url(p),
            "ok": data is not None,
            "bytes": len(data) if data else 0,
        })
    return json.dumps(out, ensure_ascii=False, indent=2)
