@echo off
REM ============================================================
REM  CourseRAG 一键启动（本机自用）
REM  唯一已验证可用的 Python 环境：evalverify venv
REM  （含 torch/torchvision/langchain/sentence-transformers 全家桶）
REM  若你自建了本地 venv，把下面 PY 改成你的 python.exe 路径即可。
REM ============================================================
setlocal
set "PY=C:\Users\user\.workbuddy\binaries\python\envs\evalverify\Scripts\python.exe"
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo [CourseRAG] 请确保 Ollama 已运行（终端执行 ollama serve），
echo            且已拉取 nomic-embed-text 与 llama3.1:8b。
echo [CourseRAG] 正在启动 Streamlit ...
"%PY%" -m streamlit run app.py
pause
