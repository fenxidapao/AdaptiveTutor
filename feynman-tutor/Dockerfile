# FeynmanTutor 生产镜像（2026-08-23 C0）
# 要点：非 root + 依赖分层安装 + SQLite 卷挂载（compose.yaml 注入 DB_PATH）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖分层安装（requirements 先 COPY，代码变更不触发重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 非 root 用户 + 数据目录（compose 卷挂载点）
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8001"]
