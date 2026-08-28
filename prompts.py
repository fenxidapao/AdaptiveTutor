"""Prompt 版本化管理（PLAN 17.2 / D3）：system prompt 移入 prompts/ 目录。

- git 可 diff、可评审（改 prompt = 一次 review 过的变更）；
- 测试可 mock 文件内容；
- 模板占位符（{logs} 等）保留 .format 语义，与原来内嵌常量完全一致。
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_cache: dict[str, str] = {}


def load(name: str) -> str:
    """读取 prompt 文件（进程内缓存）。name 形如 'diagnostic.md'。"""
    if name not in _cache:
        _cache[name] = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return _cache[name]
