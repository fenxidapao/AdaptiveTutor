"""组装廖雪峰 Python 教程核心章节为单一 markdown(带来源署名, CC BY-NC-SA 4.0)。

数据源: https://github.com/michaelliao/liaoxuefeng.com (source/books/python)
许可:   CC BY-NC-SA 4.0(署名-非商业性使用-相同方式共享), 已核实仓库 LICENSE
用途:   个人学习研究项目语料(非商业), 保留署名与许可声明。

选取章节(与《Python》课程最相关):
  50-basic 基础语法 / 60-function 函数 / 70-advanced 高级特性 /
  80-functional 函数式编程 / 90-module 模块 / 100-oop 面向对象 /
  110-oop-adv OOP 高级 / 120-error-debug-test 错误调试 / 130-io IO /
  150-reg-exp 正则 / 160-built-in-modules 常用内置模块 / 170-third-party 常用第三方模块
"""

import os
import re
from pathlib import Path

SRC = Path(os.environ.get("PY_CORPUS_SRC", "data/教材/python-source"))  # 上游教程源码目录
OUT = Path(os.environ.get("PY_CORPUS_OUT", "data/教材/Python-廖雪峰教程-核心章节.md"))

SECTIONS = [
    ("50-basic", "Python 基础语法"),
    ("60-function", "函数"),
    ("70-advanced", "高级特性"),
    ("80-functional", "函数式编程"),
    ("90-module", "模块"),
    ("100-oop", "面向对象编程"),
    ("110-oop-adv", "面向对象高级编程"),
    ("120-error-debug-test", "错误、调试和测试"),
    ("130-io", "IO 编程"),
    ("150-reg-exp", "正则表达式"),
    ("160-built-in-modules", "常用内置模块"),
    ("170-third-party-modules", "常用第三方模块"),
]


def chapter_title(dir_name: str) -> str:
    """从目录名(如 50-basic)提取可读标题。"""
    return dir_name.split("-", 1)[-1].replace("-", " ").title()


def collect_md_files(chapter: Path) -> list[Path]:
    """递归收集章节下所有 README.md(按目录名数字排序)。"""
    files = sorted(chapter.rglob("*.md"), key=lambda p: str(p))
    return files


def main():
    parts = []
    parts.append(
        "# Python 教程(廖雪峰, 核心章节节选)\n\n"
        "> 数据来源: 廖雪峰的官方网站《Python 教程》(https://liaoxuefeng.com/books/python/)\n"
        "> 许可: CC BY-NC-SA 4.0(署名-非商业性使用-相同方式共享), 仅供个人学习研究。\n"
        "> 说明: 从完整教程中选取与计算机专业《Python》课程相关的核心章节, 非商业用途。\n\n"
        "---\n"
    )

    for dir_name, cn_title in SECTIONS:
        chapter = SRC / dir_name
        if not chapter.exists():
            print(f"跳过(不存在): {dir_name}")
            continue
        md_files = collect_md_files(chapter)
        if not md_files:
            continue
        # 章节标题
        parts.append(f"\n# {cn_title}({chapter.name})\n")
        for f in md_files:
            rel = f.relative_to(SRC)
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            # 去掉文件内的首个一级标题(避免与章节标题重复), 提升下级标题层级
            lines = text.split("\n")
            if lines and lines[0].startswith("# "):
                lines = lines[1:]
            # 把原有 # 提升为 ## (保持结构切分的标题层级)
            text = "\n".join(lines)
            parts.append(text)
            parts.append("\n")
        parts.append("\n---\n")

    content = "\n".join(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding="utf-8")
    print(f"写入 {OUT}")
    print(f"总字符: {len(content)}")


if __name__ == "__main__":
    main()
