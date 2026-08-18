@echo off
REM ============================================================
REM  CourseRAG launcher (portable)
REM  Python detection order:
REM   1. .\venv\Scripts\python.exe   (local venv you created)
REM   2. %COURSERAG_PYTHON%          (env var pointing to any python.exe)
REM   3. python on PATH              (fallback, must have deps)
REM  Requires: Ollama running with nomic-embed-text + llama3.1:8b
REM  No hardcoded usernames / paths -> works on any machine.
REM ============================================================
setlocal
set "ROOT=%~dp0"

if exist "%ROOT%venv\Scripts\python.exe" (
    set "PY=%ROOT%venv\Scripts\python.exe"
    goto :check
)
if defined COURSERAG_PYTHON (
    set "PY=%COURSERAG_PYTHON%"
    goto :check
)
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :check
)

echo [CourseRAG] ERROR: no Python interpreter found.
echo   Option A: create a local venv
echo     python -m venv venv
echo     venv\Scripts\pip install -r requirements.txt
echo   Option B: set env var to your python.exe
echo     set COURSERAG_PYTHON=C:\path\to\python.exe
goto :eof

:check
REM verify the chosen python actually has the required packages
"%PY%" -c "import streamlit, ollama, chromadb" >nul 2>&1
if errorlevel 1 (
    echo [CourseRAG] ERROR: chosen Python lacks dependencies.
    echo   Interpreter: %PY%
    echo   Fix: %PY% -m pip install -r requirements.txt
    echo   Or set COURSERAG_PYTHON to a venv that already has them.
    goto :eof
)

echo [CourseRAG] Using Python: %PY%
echo [CourseRAG] Make sure Ollama is running (ollama serve) with models pulled.
echo [CourseRAG] Starting Streamlit on http://localhost:8501 ...
cd /d "%ROOT%"
"%PY%" -m streamlit run app.py --server.port 8501
pause
