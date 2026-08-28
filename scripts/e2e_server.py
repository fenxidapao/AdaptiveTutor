"""E2E 冒烟测试服务器：真实 Web 代码 + 打桩 LLM（快、免费、可重复）。

用法：/d/anacoda3/python.exe scripts/e2e_server.py   # 起 8002 端口
配套：scripts/e2e_flow.py（Playwright 6 步闭环冒烟，PLAN 18.5 D2）
原理：真实跑 web.app 全部路由与判题链路，仅替换 model.chat 为固定返回，
      按 caller 返回对应格式（诊断 JSON 画像/追问一句话/盲点 JSON/推荐 JSON…）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402


def _fake_chat(messages, temperature=0.3, max_tokens=2000, caller=None, **kw):
    c = caller or ""
    if "diagnostic" in c:
        return ('{"weak_points":[{"kp_id":"python.list.slice","reason":"做错切片题",'
                '"evidence":["py.ch1.d.1"]}],"learning_style":"简答","avg_correct":0.5}')
    if "gaps" in c:
        return '{"gaps":["切片 end 不含在结果里"]}'
    if "rag" in c:
        return '{"used":[0],"explanation":"列表切片用冒号取子集，start 含 end 不含，步长默认 1。"}'
    if "recommender" in c:
        return '{"reasons":{"py.ch1.d.1":"补切片薄弱点"}}'
    if "planner" in c:
        return "按前置依赖先学基础，再优先攻克薄弱点列表切片。"
    if "hint" in c:
        return "想想切片 end 是否含在结果里。"
    return "很好，再想想边界情况？"  # feynman followup / 兜底


model.chat = _fake_chat
model.chat_with_fallback = _fake_chat

import uvicorn  # noqa: E402
from web.app import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
