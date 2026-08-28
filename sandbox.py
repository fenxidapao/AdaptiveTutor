"""沙箱执行器：subprocess 跑学生代码（PLAN 14.3）。

安全约束（PLAN 11 风险对策）：
. 超时 5 秒（死循环防护）
. stdout/stderr 各限 64KB
. 工作目录为临时目录（学生代码无写权限于项目目录）
. Windows 下 CREATE_NO_WINDOW（不弹黑窗口）

返回 {'ok':bool,'stdout':str,'stderr':str,'exit_code':int,'timed_out':bool}
"""

import os
import subprocess
import sys
import tempfile

import config


def run_python(code: str, timeout: int | None = None,
               max_output: int | None = None) -> dict:
    """在临时目录跑一段 Python 代码。code 为空视为非法输入。

    timeout 默认 config.SANDBOX_TIMEOUT(5s)；max_output 默认 64KB。
    """
    timeout = timeout or config.SANDBOX_TIMEOUT
    max_output = max_output or config.SANDBOX_MAX_OUTPUT

    if not code or not code.strip():
        return {"ok": False, "stdout": "", "stderr": "代码为空",
                "exit_code": -1, "timed_out": False}

    # 富文本粘贴常带前导 BOM（\ufeff）：strip() 不去它，_python -c_ 会报语法错，需剥掉
    code = code.lstrip("\ufeff")

    # 子进程标志：Windows 不弹控制台窗口
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    with tempfile.TemporaryDirectory(prefix="feynman_sandbox_") as tmpdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=timeout,
                cwd=tmpdir,
                creationflags=creationflags,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return {
                "ok": proc.returncode == 0,
                "stdout": stdout[:max_output],
                "stderr": stderr[:max_output],
                "exit_code": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "stdout": (e.stdout or b"").decode("utf-8", errors="replace")[:max_output],
                "stderr": f"[超时] 代码运行超过 {timeout} 秒，已终止（可能是死循环）。",
                "exit_code": -2,
                "timed_out": True,
            }


def check_infrastructure() -> None:
    """启动自检：沙箱可运行吗？（main.py --health 用）"""
    r = run_python("print(1+1)")
    assert r["ok"] and r["stdout"].strip() == "2", f"沙箱自检失败: {r}"
