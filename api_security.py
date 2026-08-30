"""
api_security.py — 多人预留扩展点（Gate 2，2026-08-24）

默认全部关闭（AUTH_ENABLED=false / RATE_LIMIT=0 / CORS_ORIGINS=*），不启用任何鉴权 / 限流，
保持本地 / 局域网部署的零摩擦。代码已就位：将来要并发 Web 服务，只需在 .env 开开关，
不必重写后端。

- API Key 鉴权：AUTH_ENABLED=true 时，所有端点要求请求头 `X-API-Key: <API_KEY>`。
- 限流：RATE_LIMIT>0 时，按客户端 IP 限制每分钟请求数（内存滑动窗口，单进程够用；
  多副本需换 Redis，见文档路线 β/ε）。
- CORS：允许源从 config.CORS_ORIGINS 读取（默认 "*"，生产收紧为具体域名）。

无新重依赖：仅 fastapi / starlette（已在 requirements-api.txt）。
"""
import secrets
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import config


def require_auth(request: Request):
    """全局鉴权依赖（FastAPI `dependencies=[Depends(require_auth)]` 注入）。
    关闭时直接放行；开启时校验 `X-API-Key` 请求头。"""
    if not config.AUTH_ENABLED:
        return
    key = request.headers.get("X-API-Key") or ""
    # 常量时间比较防时序侧信道（评审 P3）；转 bytes 规避 compare_digest 的 ASCII str 限制
    if not config.API_KEY or not secrets.compare_digest(
        key.encode("utf-8"), config.API_KEY.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权：请提供正确的 X-API-Key（或设 AUTH_ENABLED=false 关闭鉴权）",
        )


# 限流器追踪的 IP 数上限（评审 P3）：恶意/扫描流量可伪造海量源 IP，把字典撑到无限大。
# 超限按 LRU 逐出「最久未活跃」的 IP（其窗口本就只在 60 秒滑动窗内有意义）。
_MAX_TRACKED_IPS = 10_000


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按 IP 滑动窗口限流（内存实现，单进程）。RATE_LIMIT=0 时不限。"""

    def __init__(self, app, per_minute: int = 0):
        super().__init__(app)
        self.per_minute = per_minute
        # OrderedDict 便于 LRU 逐出：value 仍为该 IP 的 60s 滑动窗口命中时间队列
        self._hits: OrderedDict[str, deque] = OrderedDict()

    async def dispatch(self, request: Request, call_next):
        if self.per_minute <= 0:
            return await call_next(request)
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._hits.get(ip)
        if window is None:
            if len(self._hits) >= _MAX_TRACKED_IPS:
                self._hits.popitem(last=False)  # 逐出最久未活跃的 IP，表容量有界
            window = self._hits[ip] = deque()
        self._hits.move_to_end(ip)  # 刷新活跃度（LRU 语义）
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= self.per_minute:
            # 注意：BaseHTTPMiddleware.dispatch 里 raise HTTPException 不会被 FastAPI
            # 异常处理器捕获，会变成 ASGI 异常 → 500（2026-08-24 实测）。必须返回
            # JSONResponse(429) 而不是 raise。
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"请求过于频繁，限制 {self.per_minute} 次/分钟（per IP）。"},
            )
        window.append(now)
        return await call_next(request)
