"""安全模块：五层纵深防御（F 阶段，PLAN 22.1 安全行）。

行业教训：防御不能赌"这一层够了"，只能假设每一层都会漏——分层的意义是让
攻击路径必须同时穿过所有层；某层被绕过时其余层仍在（本文作者踩过的坑：
"以为其他层能顶上，结果攻击路径根本不是我想的那一条"）。

五层与本项目落点（L4 业务校验大部分已在既有代码中，这里形式化归位）：

  L1 Prompt 边界      screen_prompt_injection(): 用户输入注入/越狱筛查——
                      高危(提示词窃取/角色越狱)直接 400，可疑(忽略指令类)放行
                      但审计，由教练 system prompt 层兜底（教学场景误杀成本高）；
                      context.sanitize_transcript() 在 LLM 边界做长度/条数硬截断。
  L2 Schema 约束+幂等 validate_grade_payload()/validate_submit_payload():
                      FastAPI 之外的业务级二次约束（类型/大小/格式）；
                      /api/grade 支持客户端 request_id 幂等（db.idempotency_*），
                      网络重试/双击不再重复计分。
  L3 风险分级审批     RISK_TIERS + is_admin(): 读=T0 / 练习写=T1 / 测评与画像写=T2 /
                      管理操作=T3（仅 ADMIN_USER_IDS），高危操作显式授权。
  L4 业务校验         判题纯规则（grader/sql_grader）+ 规则画像（E5）+ mode 服务端
                      强制（C1）+ 掌握门槛（blocked）——业务正确性不依赖 LLM 自觉。
  L5 全链路审计       db.audit_logs: 注入命中/鉴权失败/幂等命中/管理访问/测评提交
                      全落审计，事件可回放攻击路径（llm_logs 补 token/耗时维度）。

设计取舍：不引入 WAF/外部网关——单机教育场景，五层的价值在"分层假设"，
不在堆组件。每一层都是可单测的纯函数或一条 SQL。
"""

import json
import re

import config
import db

# ==================== L1 Prompt 边界：注入筛查 ====================

# (编译正则, 类别名)。高危=直接操纵模型身份/窃取系统提示 → 拒绝；
# 可疑=指令覆盖类 → 放行+审计（教练 system prompt 已有防线，误杀成本高）
_INJECTION_PATTERNS = [
    (re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s|"
        r"disregard\s+(all\s+)?(previous|prior|above)\s|"
        r"forget\s+(all\s+)?(previous|prior|above)\s", re.I), "忽略指令", 1),
    (re.compile(
        r"忽略(掉?)(之前|以上|上面|前面|此前)的(所有)?(指令|设定|提示|要求|规则)|"
        r"无视(之前|以上|上面|此前)的(所有)?(指令|设定|提示|要求|规则)"), "忽略指令", 1),
    (re.compile(
        r"(reveal|show|print|output|输出|打印|告诉我).{0,16}"
        r"(system\s*prompt|system\s*message|系统提示|系统指令|初始提示|"
        r"你的(初始|系统|隐藏)?(指令|提示词|设定))", re.I), "提示词窃取", 2),
    (re.compile(
        r"(system\s*prompt|系统提示词?|系统指令|初始(提示|指令)|隐藏提示).{0,20}"
        r"(输出|打印|复述|repeat|原样|发给我|给我看|一字不差)", re.I), "提示词窃取", 2),
    (re.compile(
        r"(you\s+are\s+now|从现在开始你是|现在你是|假装你是|扮演).{0,24}"
        r"(developer|admin|root|unrestricted|无限制|不受限|没有限制|DAN)", re.I),
     "角色越狱", 2),
    (re.compile(r"developer\s*mode|\bDAN\s*mode\b|越狱模式|jailbreak", re.I),
     "越狱模式", 2),
]

# 注入筛查覆盖的输入字段内容长度上限（防正则回溯/超大输入拖垮匹配）
_SCREEN_TEXT_LIMIT = 20000


def screen_prompt_injection(text: str) -> dict:
    """L1 Prompt 边界：筛查用户输入中的注入/越狱意图。

    返回 {"risk": 0|1|2, "matched": [类别名...]}：
      risk 0 = 干净；risk 1 = 可疑（忽略指令类，放行但必须审计，
      由教练 system prompt + hint rubric 双层兜底）；
      risk 2 = 高危（提示词窃取/角色越狱/越狱模式，调用方应直接 400）。
    纯函数可单测；正则只做意图特征匹配，宁漏勿误杀（教学场景）。
    """
    text = str(text or "")[:_SCREEN_TEXT_LIMIT]
    matched: list[str] = []
    risk = 0
    for pat, name, tier in _INJECTION_PATTERNS:
        if pat.search(text):
            matched.append(name)
            risk = max(risk, tier)
    return {"risk": risk, "matched": matched}


def screen_transcript_injection(transcript) -> dict:
    """对整段 transcript（多轮消息）做注入筛查（feynman 端点入口用）。"""
    if not isinstance(transcript, list):
        return {"risk": 0, "matched": []}
    joined = " ".join(
        str(m.get("content", "")) for m in transcript if isinstance(m, dict))
    return screen_prompt_injection(joined)


# ==================== L2 Schema 约束 + 幂等 ====================

_MAX_TEXT_FIELD = 4000          # 自由文本字段上限（字符）
_MAX_ANSWER_ENTRIES = 100       # 整套提交 answers 键数上限
_MAX_LIST_FIELD = 50            # transcript 等列表字段长度上限


def _check_id_field(value, name: str, errs: list[str]) -> str:
    """id 类字段通用校验：非空、≤64 字符、无控制字符/空白。"""
    s = str(value or "").strip()
    if not s:
        errs.append(f"{name} 必填")
    elif len(s) > 64 or any(ord(ch) < 32 for ch in s):
        errs.append(f"{name} 含非法字符或超长")
    return s


def validate_grade_payload(payload: dict) -> list[str]:
    """L2：/api/grade 二次约束（FastAPI 已保证顶层是 JSON 对象）。"""
    errs: list[str] = []
    _check_id_field(payload.get("user_id"), "user_id", errs)
    _check_id_field(payload.get("ex_id"), "ex_id", errs)
    answer = payload.get("answer", "")
    if len(str(answer)) > _MAX_TEXT_FIELD:
        errs.append("answer 超长")
    request_id = payload.get("request_id")
    if request_id is not None and (not str(request_id).strip()
                                   or len(str(request_id)) > 128):
        errs.append("request_id 非法")
    return errs


def validate_submit_payload(payload: dict) -> list[str]:
    """L2：/api/quiz/{kind}/submit 二次约束（answers 结构/大小）。"""
    errs: list[str] = []
    _check_id_field(payload.get("user_id"), "user_id", errs)
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        errs.append("answers 必须是对象")
        return errs
    if len(answers) > _MAX_ANSWER_ENTRIES:
        errs.append(f"answers 条目过多（>{_MAX_ANSWER_ENTRIES}）")
    for k, v in list(answers.items())[:_MAX_ANSWER_ENTRIES + 1]:
        if len(str(k)) > 64:
            errs.append("answers 键非法")
            break
        if len(str(v)) > _MAX_TEXT_FIELD:
            errs.append("answers 值超长")
            break
    elapsed = payload.get("elapsed_seconds")
    if elapsed is not None and not isinstance(elapsed, (int, float)):
        errs.append("elapsed_seconds 必须是数字")
    return errs


def validate_transcript_payload(transcript) -> list[str]:
    """L2：/api/feynman/* transcript 结构约束（内容消毒由 context 层做）。"""
    errs: list[str] = []
    if not isinstance(transcript, list):
        errs.append("transcript 必须是数组")
        return errs
    if len(transcript) > _MAX_LIST_FIELD:
        errs.append(f"transcript 条数过多（>{_MAX_LIST_FIELD}）")
    for m in transcript[:_MAX_LIST_FIELD + 1]:
        if (not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                or not isinstance(m.get("content"), str)):
            errs.append("transcript 元素必须是 {role: user|assistant, content: str}")
            break
    return errs


def idempotent_response(request_id, user_id: str,
                        db_path: str | None = None) -> dict | None:
    """L2 幂等：request_id 已见过 → 返回首次响应（重放不再计分）。

    request_id 为空/未启用 → None（走正常流程）。显式客户端键而不是
    "同答案 N 秒去重"：学生连续两次故意提交同一答案是合法学习行为
    （E2 策略升级依赖连续计分），不能吞。
    """
    if not request_id:
        return None
    row = db.idempotency_get(str(request_id), db_path)
    if row is None:
        return None
    try:
        return json.loads(row["response"])
    except (ValueError, TypeError):
        return None


def store_idempotent(request_id, user_id: str, response: dict,
                     db_path: str | None = None) -> None:
    """L2 幂等：记录 request_id → 首次响应（响应存 JSON 字符串）。"""
    if not request_id:
        return
    db.idempotency_put(str(request_id), user_id,
                       json.dumps(response, ensure_ascii=False), db_path)


# ==================== L3 风险分级 + 管理面 ====================

# 风险分级表：每类写操作的风险档位（审计 risk 字段与审批策略的依据）
RISK_TIERS = {
    "read": 0,              # 纯查询（报告/热力图/队列）
    "practice_write": 1,    # 练习判题写日志（可重复、可追溯，自动放行）
    "assessment_write": 2,  # 测评提交/画像重写（影响实验数据与分流，自动放行+强制审计）
    "profile_write": 2,
    "admin": 3,             # 管理操作（用户列表/审计查看）：仅 ADMIN_USER_IDS
}


def risk_classify(event: str) -> int:
    """事件 → 风险档位（未知事件按 T2 保守处理）。"""
    return RISK_TIERS.get(event, 2)


def is_admin(user_id: str) -> bool:
    """L3 管理面授权：user_id ∈ ADMIN_USER_IDS。"""
    return bool(user_id) and user_id in config.ADMIN_USER_IDS


# ==================== L5 全链路审计 ====================

def audit(event: str, user_id: str | None = None, session_id: str | None = None,
          risk: int | None = None, detail: dict | None = None,
          db_path: str | None = None) -> None:
    """审计包装：未显式给 risk 时按风险分级表推断（未知事件保守 T2）。"""
    db.audit_log(event, user_id=user_id, session_id=session_id,
                 risk=RISK_TIERS.get(event, 2) if risk is None else risk,
                 detail=detail, db_path=db_path)
