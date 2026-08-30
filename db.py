"""状态库封装：SQLite，全部表带 user_id（多用户，PLAN 6.2）。

0 号用户 = 本人（'u0'）；后续同学/网上用户各自一个 user_id，数据结构天然支持，
从 P0 第一行代码就是多用户，不做返工。
"""

import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT,
  group_name TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_points (
  user_id TEXT NOT NULL,
  kp_id TEXT NOT NULL,
  title TEXT NOT NULL,
  chapter TEXT NOT NULL,
  prerequisites TEXT DEFAULT '[]',
  mastery REAL DEFAULT 0.0,
  seen_count INTEGER DEFAULT 0,
  correct_count INTEGER DEFAULT 0,
  explain_count INTEGER DEFAULT 0,
  last_explained TEXT,
  last_practiced TEXT,
  next_review TEXT,
  status TEXT DEFAULT 'new',
  review_interval INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, kp_id)
);

CREATE TABLE IF NOT EXISTS exercise_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  kp_id TEXT,
  ex_id TEXT,
  attempt INTEGER,
  correct INTEGER,
  user_answer TEXT,
  feedback TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  chapter TEXT,
  kind TEXT,
  mode TEXT,
  score REAL,
  total INTEGER,
  elapsed_seconds REAL,     -- 整套答题耗时（作弊检测依据，C1）
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS profile (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  weak_points TEXT,
  learning_style TEXT,
  avg_correct REAL,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS llm_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  caller TEXT,                -- 调用环节：diagnostic/feynman/hint/planner/recommender/rag/assessor
  model TEXT,                 -- 实际模型名
  prompt_tokens INTEGER,      -- 输入 token
  completion_tokens INTEGER,  -- 输出 token
  total_tokens INTEGER,       -- 总 token
  latency_ms INTEGER,         -- 单次调用耗时（毫秒）
  attempts INTEGER,           -- 外层重试次数
  expanded INTEGER,           -- 是否发生推理预算扩容
  source TEXT,                -- api / ollama
  user_id TEXT,               -- 发起用户（C 阶段配额按用户计费）
  session_id TEXT,            -- 发起会话（D4 session trace：一次闭环可回放）
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS reflow_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  chapter TEXT,
  round INTEGER DEFAULT 1,          -- 回流轮次（1 起，上限 REFLOW_MAX_ROUNDS）
  trigger_score REAL,               -- 触发回流时的后测正确率
  weak_kps TEXT,                    -- JSON：薄弱点 kp_id 列表（回流学习任务）
  status TEXT DEFAULT 'open',       -- open 进行中 / completed 重测达标 / given_up 超轮放弃
  retest_score REAL,                -- 重测后测正确率（外部验证）
  created_at TEXT,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_logs (   -- F 阶段安全 L5：全链路审计（只记安全相关事件）
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  session_id TEXT,
  event TEXT NOT NULL,              -- injection_block/injection_flag/auth_fail/grade_idempotent_hit/
                                    -- assessment_submit/users_list/audit_view/audit_denied ...
  risk INTEGER DEFAULT 0,           -- 风险分级 T0-T3（security.RISK_TIERS）
  detail TEXT,                      -- JSON 详情
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS idempotency_keys (  -- F 阶段安全 L2：写操作幂等（网络重试/双击不重复计分）
  req_key TEXT PRIMARY KEY,         -- 客户端生成的 request_id
  user_id TEXT,
  response TEXT,                    -- 首次响应 JSON（重放时原样返回）
  created_at TEXT
);
"""


def _conn(db_path: str | None = None) -> sqlite3.Connection:
    """新建连接（F 阶段加固：WAL + busy_timeout）。

    WAL：读写不互斥、写写由 busy_timeout 排队——多用户并发答题不再
    "database locked"（此前每操作新建裸连接，无任何并发配置）。
    busy_timeout=5000：写锁被占时等 5s 而不是立即抛错（会话级"锁"的
    SQLite 等价物：靠引擎排队，不靠应用层分布式锁）。
    journal_mode 是库文件级持久属性，重复设置无害（已是 WAL 时为空操作）。
    """
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: str | None = None) -> None:
    with _conn(db_path) as c:
        c.executescript(SCHEMA)
        # 迁移：老库缺列时补（IF NOT EXISTS 不会为已有表加列）
        ucols = {r[1] for r in c.execute("PRAGMA table_info(users)")}
        if "group_name" not in ucols:
            c.execute("ALTER TABLE users ADD COLUMN group_name TEXT DEFAULT NULL")
        if "password_hash" not in ucols:
            c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT NULL")
        kcols = {r[1] for r in c.execute("PRAGMA table_info(knowledge_points)")}
        if "review_interval" not in kcols:
            c.execute("ALTER TABLE knowledge_points ADD COLUMN review_interval INTEGER DEFAULT 0")
        acols = {r[1] for r in c.execute("PRAGMA table_info(assessments)")}
        if "elapsed_seconds" not in acols:
            c.execute("ALTER TABLE assessments ADD COLUMN elapsed_seconds REAL")
        lcols = {r[1] for r in c.execute("PRAGMA table_info(llm_logs)")}
        if "user_id" not in lcols:
            c.execute("ALTER TABLE llm_logs ADD COLUMN user_id TEXT")
        if "session_id" not in lcols:
            c.execute("ALTER TABLE llm_logs ADD COLUMN session_id TEXT")


def get_user(user_id: str, name: str | None = None, db_path: str | None = None) -> dict:
    """用户存在则返回，不存在则创建。返回 {user_id, name, created_at, group_name}。

    传了 name 且用户已存在时更新 name（注册场景补全用户名）。
    """
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            now = datetime.now().isoformat(timespec="seconds")
            c.execute(
                "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name or user_id, now),
            )
            row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        elif name and name != row["name"]:
            c.execute("UPDATE users SET name=? WHERE user_id=?", (name, user_id))
            row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row)


# ==================== C 阶段：注册 / 会话 / 配额（PLAN 18.3） ====================


def _hash_password(pwd: str) -> str:
    """pbkdf2 加盐哈希，格式 salt$hash（标准库，零依赖）。"""
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${h}"


def _verify_password(pwd: str, stored: str) -> bool:
    if not stored:
        return False
    try:
        salt, h = stored.split("$", 1)
    except ValueError:
        return False
    return h == hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000).hex()


def _auto_group(db_path: str | None = None) -> str:
    """注册自动分组：分到当前人数少的组（均衡分配，实验设计见 docs/EXPERIMENT.md）。

    只统计有 password_hash 的注册账号——遗留无凭据账号（u0/smoke_test 等
    get_user 隐式创建，永远无法登录）不占用均衡计数，避免污染分组比例。
    """
    with _conn(db_path) as c:
        feynman = c.execute(
            "SELECT COUNT(*) FROM users WHERE group_name='feynman' "
            "AND password_hash IS NOT NULL").fetchone()[0]
        lecture = c.execute(
            "SELECT COUNT(*) FROM users WHERE group_name='lecture' "
            "AND password_hash IS NOT NULL").fetchone()[0]
    return "lecture" if lecture < feynman else "feynman"


def register_user(user_id: str, password: str, name: str | None = None,
                  group_name: str | None = None,
                  db_path: str | None = None) -> dict:
    """注册新用户：密码哈希 + 自动分组。user_id 已存在则抛 ValueError。"""
    with _conn(db_path) as c:
        row = c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row is not None:
            raise ValueError(f"user_id 已存在: {user_id}")
        now = datetime.now().isoformat(timespec="seconds")
        g = group_name or _auto_group(db_path)
        c.execute(
            "INSERT INTO users (user_id, name, created_at, group_name, password_hash) "
            "VALUES (?,?,?,?,?)",
            (user_id, name or user_id, now, g, _hash_password(password)),
        )
        row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row)


def verify_user(user_id: str, password: str, db_path: str | None = None) -> bool:
    with _conn(db_path) as c:
        row = c.execute("SELECT password_hash FROM users WHERE user_id=?", (user_id,)).fetchone()
    return bool(row) and _verify_password(password, row["password_hash"])


def create_session(user_id: str, db_path: str | None = None) -> str:
    """登录建 session（uuid4，防串台）。返回 session_id。"""
    sid = uuid.uuid4().hex
    with _conn(db_path) as c:
        c.execute("INSERT INTO sessions (session_id, user_id, created_at) VALUES (?,?,?)",
                  (sid, user_id, datetime.now().isoformat(timespec="seconds")))
    return sid


def get_session_user(session_id: str, db_path: str | None = None) -> str | None:
    with _conn(db_path) as c:
        row = c.execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return row["user_id"] if row else None


def delete_session(session_id: str, db_path: str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


def today_usage(user_id: str, db_path: str | None = None) -> int:
    """该用户今日 LLM 调用次数（llm_logs 计数，配额依据）。"""
    today = datetime.now().date().isoformat()
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) FROM llm_logs WHERE user_id=? AND date(created_at)=?",
            (user_id, today)).fetchone()
    return row[0]


def global_today_usage(db_path: str | None = None) -> int:
    today = datetime.now().date().isoformat()
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) FROM llm_logs WHERE date(created_at)=?", (today,)).fetchone()
    return row[0]


def quota_exceeded(user_id: str, db_path: str | None = None) -> tuple[bool, str]:
    """配额检查：每用户上限 + 全局熔断。返回 (是否超限, 文案)。"""
    if today_usage(user_id, db_path) >= config.DAILY_LLM_LIMIT_PER_USER:
        return True, f"今日学习额度已用完（{config.DAILY_LLM_LIMIT_PER_USER} 次），明天再来吧"
    if global_today_usage(db_path) >= config.GLOBAL_DAILY_LLM_LIMIT:
        return True, "今日全站额度已满，请明天再来"
    return False, ""


def assign_group(user_id: str, group_name: str, db_path: str | None = None) -> dict:
    """把用户分到实验组（feynman / lecture）。返回更新后的用户。"""
    get_user(user_id, db_path=db_path)  # 确保存在
    with _conn(db_path) as c:
        c.execute("UPDATE users SET group_name=? WHERE user_id=?",
                  (group_name, user_id))
        row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        return dict(row)


def list_users(db_path: str | None = None) -> list[dict]:
    """全部用户（含分组），供 P1+ 组间对照实验统计。"""
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]


def list_assessments(user_id: str, chapter: str | None = None,
                     db_path: str | None = None) -> list[dict]:
    """某用户的全部测评记录（按时间升序）。C1 实验分析与 auth 测试共用。"""
    sql = "SELECT * FROM assessments WHERE user_id=?"
    args: list = [user_id]
    if chapter:
        sql += " AND chapter=?"
        args.append(chapter)
    sql += " ORDER BY created_at"
    with _conn(db_path) as c:
        rows = c.execute(sql, args).fetchall()
        return [dict(r) for r in rows]


def upsert_kp(user_id: str, kp: dict, db_path: str | None = None) -> None:
    """更新知识点状态（不存在则插入）。kp 至少含 kp_id/title/chapter。"""
    fields = (
        "kp_id", "title", "chapter", "prerequisites", "mastery", "seen_count",
        "correct_count", "explain_count", "last_explained", "last_practiced",
        "next_review", "status",
    )
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM knowledge_points WHERE user_id=? AND kp_id=?",
            (user_id, kp["kp_id"]),
        ).fetchone()
        if row is None:
            defaults = {
                "prerequisites": "[]", "mastery": 0.0, "seen_count": 0,
                "correct_count": 0, "explain_count": 0, "last_explained": None,
                "last_practiced": None, "next_review": None, "status": "new",
            }
            merged = {**defaults, **kp}
            cols = ("user_id",) + fields
            c.execute(
                f"INSERT INTO knowledge_points ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})",
                (user_id, *(merged.get(f) for f in fields)),
            )
        else:
            updates = {f: kp[f] for f in fields if f in kp}
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                c.execute(
                    f"UPDATE knowledge_points SET {sets} WHERE user_id=? AND kp_id=?",
                    (*updates.values(), user_id, kp["kp_id"]),
                )


def log_exercise(user_id: str, ex_id: str, kp_id: str, correct: bool,
                 user_answer: str, feedback: str, attempt: int = 1,
                 db_path: str | None = None) -> None:
    """记答题日志，并同步更新 knowledge_points 掌握度（P3 修复：判题链路断点）。

    mastery = correct_count / seen_count（最近 50 条内该 kp 的正确率）；
    status: 无记录 new / mastery>=0.8 mastered / >0 learning / 全错 reviewing。
    事务提交后再做动态画像增量更新（E5 Memory，PLAN 可扩展点 3）。
    """
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO exercise_logs (user_id, kp_id, ex_id, attempt, correct, "
            "user_answer, feedback, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, kp_id, ex_id, attempt, int(correct), user_answer, feedback,
             datetime.now().isoformat(timespec="seconds")),
        )
        _update_kp_mastery(c, user_id, kp_id, correct)
    # 事务已提交（mastery 已更新），增量维护画像——必须在本连接 commit 之后，
    # 否则第二个连接读不到未提交的 mastery，答对移除判断会失准
    update_profile_incremental(user_id, kp_id, correct, ex_id, db_path)


def update_profile_incremental(user_id: str, kp_id: str, correct: bool,
                               ex_id: str, db_path: str | None = None) -> None:
    """动态画像（E5 Memory / PLAN 可扩展点 3）：答题后增量维护 weak_points。

    纯规则，不调 LLM（防漂移）：
    - 答错：该 kp 不在 weak_points → 追加 {kp_id, reason:"增量：答题答错",
      evidence:[ex_id]}；已存在 → evidence 追加 ex_id（去重，最多 10 条）；
    - 答对且该 kp mastery>=0.8：从 weak_points 移除（薄弱点消除，证据链作废）。
    写新 profile 行（保持历史可追溯，get_profile 取最新）。
    """
    if not kp_id:
        return
    prof = get_profile(user_id, db_path)
    weak = parse_weak_details(prof)
    state = get_kp(user_id, kp_id, db_path) or {}
    mastery = state.get("mastery") or 0.0

    changed = False
    if not correct:
        entry = next((w for w in weak if w["kp_id"] == kp_id), None)
        if entry is None:
            weak.append({"kp_id": kp_id, "reason": "增量：答题答错", "evidence": [ex_id]})
            changed = True
        else:
            ev = list(entry.get("evidence") or [])
            if ex_id and ex_id not in ev:
                ev = (ev + [ex_id])[:10]
                entry["evidence"] = ev
                changed = True
    elif mastery >= 0.8:
        before = len(weak)
        weak = [w for w in weak if w["kp_id"] != kp_id]
        changed = len(weak) != before

    # 有画像才写（首次答题且答错也会建画像：weak 非空）；答对但无画像不建空画像
    if changed and (prof is not None or weak):
        save_profile(user_id, {
            "weak_points": weak,
            "learning_style": (prof or {}).get("learning_style", "简答"),
            "avg_correct": (prof or {}).get("avg_correct", 0.0),
        }, db_path)


def _update_kp_mastery(c: sqlite3.Connection, user_id: str, kp_id: str,
                       correct: bool) -> None:
    """内部：更新 knowledge_points 的掌握度（log_exercise 后调用）。

    只统计最近 50 条答题（滑动窗口），防止远古数据拖累当前掌握度。
    status: 无记录 new / 最近 3 条全错 blocked / mastery>=0.8 mastered /
            >0 learning / 全错 reviewing（掌握门槛：卡住不放行）。
    """
    if not kp_id:
        return
    row = c.execute(
        "SELECT COUNT(*) n, SUM(correct) ok FROM ("
        "  SELECT correct FROM exercise_logs WHERE user_id=? AND kp_id=? "
        "  ORDER BY id DESC LIMIT 50)",
        (user_id, kp_id),
    ).fetchone()
    seen = row["n"] or 0
    ok = row["ok"] or 0
    mastery = round(ok / seen, 2) if seen else 0.0
    # 掌握门槛：最近 3 条答题全错 → blocked（卡住不放行，依赖它的 kp 暂缓）
    recent = [r["correct"] for r in c.execute(
        "SELECT correct FROM exercise_logs WHERE user_id=? AND kp_id=? "
        "ORDER BY id DESC LIMIT 3", (user_id, kp_id))]
    blocked = len(recent) >= 3 and all(x == 0 for x in recent)
    if blocked:
        status = "blocked"
    elif mastery >= 0.8:
        status = "mastered"
    elif mastery > 0:
        status = "learning"
    else:
        status = "reviewing"
    now = datetime.now().isoformat(timespec="seconds")
    c.execute(
        "UPDATE knowledge_points SET mastery=?, seen_count=?, correct_count=?, "
        "status=?, last_practiced=? WHERE user_id=? AND kp_id=?",
        (mastery, seen, ok, status, now, user_id, kp_id),
    )
    # 该 kp 还没初始化过（无 title 等元数据）→ 由 upsert_kp 补全；这里仅更新计数
    if c.execute("SELECT 1 FROM knowledge_points WHERE user_id=? AND kp_id=?",
                 (user_id, kp_id)).fetchone() is None:
        c.execute(
            "INSERT OR IGNORE INTO knowledge_points (user_id, kp_id, title, chapter, "
            "mastery, seen_count, correct_count, status, last_practiced) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, kp_id, kp_id, "", mastery, seen, ok, status, now),
        )


def record_assessment(user_id: str, chapter: str, kind: str, mode: str,
                      score: float, total: int, elapsed_seconds: float | None = None,
                      db_path: str | None = None) -> None:
    """记录前测/后测结果。elapsed_seconds: 整套答题耗时（作弊检测依据，C1）。"""
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO assessments (user_id, chapter, kind, mode, score, total, "
            "elapsed_seconds, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, chapter, kind, mode, score, total, elapsed_seconds,
             datetime.now().isoformat(timespec="seconds")),
        )


def get_kp(user_id: str, kp_id: str, db_path: str | None = None) -> dict | None:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM knowledge_points WHERE user_id=? AND kp_id=?",
            (user_id, kp_id),
        ).fetchone()
        return dict(row) if row else None


def list_kps(user_id: str, db_path: str | None = None) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM knowledge_points WHERE user_id=? ORDER BY chapter, kp_id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_profile(user_id: str, db_path: str | None = None) -> dict | None:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM profile WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def parse_weak_ids(profile: dict | None) -> list[str]:
    """从 profile 提取薄弱点 kp_id 列表。

    兼容两种存储格式：
    - 旧格式：["kp_id", ...]（字符串列表）
    - 新格式：[{"kp_id": "...", "reason": "...", "evidence": ["ex_id", ...]}, ...]
    """
    if not profile:
        return []
    raw = profile.get("weak_points")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    ids = []
    for item in raw:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("kp_id"):
            ids.append(item["kp_id"])
    return ids


def parse_weak_details(profile: dict | None) -> list[dict]:
    """从 profile 提取薄弱点完整结构（含 reason/evidence），兼容旧格式自动补 reason。"""
    if not profile:
        return []
    raw = profile.get("weak_points")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    details = []
    for item in raw:
        if isinstance(item, str):
            details.append({"kp_id": item, "reason": "", "evidence": []})
        elif isinstance(item, dict) and item.get("kp_id"):
            details.append(item)
    return details


def save_profile(user_id: str, profile: dict, db_path: str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO profile (user_id, weak_points, learning_style, avg_correct, updated_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, json.dumps(profile.get("weak_points", []), ensure_ascii=False),
             profile.get("learning_style", ""), profile.get("avg_correct", 0.0),
             datetime.now().isoformat(timespec="seconds")),
        )


def get_assessments(user_id: str, chapter: str | None = None,
                    db_path: str | None = None) -> list[dict]:
    with _conn(db_path) as c:
        if chapter:
            rows = c.execute(
                "SELECT * FROM assessments WHERE user_id=? AND chapter=? ORDER BY id",
                (user_id, chapter),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM assessments WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_exercise_logs(user_id: str, kp_id: str | None = None,
                      db_path: str | None = None) -> list[dict]:
    with _conn(db_path) as c:
        if kp_id:
            rows = c.execute(
                "SELECT * FROM exercise_logs WHERE user_id=? AND kp_id=? ORDER BY id",
                (user_id, kp_id),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM exercise_logs WHERE user_id=? ORDER BY id", (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def review_due(user_id: str, db_path: str | None = None) -> list[dict]:
    """到期复习的知识点（next_review 非空且 <= 今天）。P2 复习闭环用，接口先备好。"""
    now = datetime.now().date().isoformat()
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM knowledge_points WHERE user_id=? AND next_review IS NOT NULL "
            "AND next_review <= ?",
            (user_id, now),
        ).fetchall()
        return [dict(r) for r in rows]


def schedule_next_review(user_id: str, kp_id: str, days: int,
                         db_path: str | None = None) -> None:
    """SM-2 简化版：间隔天数后复习。P0 先记字段，P2 做闭环。"""
    next_review = (datetime.now() + timedelta(days=days)).date().isoformat()
    upsert_kp(user_id, {"kp_id": kp_id, "next_review": next_review}, db_path)


# ==================== E1 学习回流记录（PLAN 20.2） ====================

def open_reflow(user_id: str, chapter: str, trigger_score: float,
                weak_kps: list[str], round_no: int = 1,
                db_path: str | None = None) -> dict:
    """创建回流任务（status=open）。返回该记录 dict。

    round_no 从 1 起；连续不达标时由调用方递增。weak_kps 是回流学习任务清单。
    """
    with _conn(db_path) as c:
        now = datetime.now().isoformat(timespec="seconds")
        cur = c.execute(
            "INSERT INTO reflow_logs (user_id, chapter, round, trigger_score, "
            "weak_kps, status, created_at) VALUES (?,?,?,?,?,'open',?)",
            (user_id, chapter, round_no, trigger_score,
             json.dumps(weak_kps, ensure_ascii=False), now),
        )
        row = c.execute("SELECT * FROM reflow_logs WHERE id=?",
                        (cur.lastrowid,)).fetchone()
        return dict(row)


def get_open_reflow(user_id: str, chapter: str | None = None,
                    db_path: str | None = None) -> dict | None:
    """最近一条进行中的回流任务（重测提交时判定"这次是重测"的依据）。"""
    sql = "SELECT * FROM reflow_logs WHERE user_id=? AND status='open'"
    args: list = [user_id]
    if chapter:
        sql += " AND chapter=?"
        args.append(chapter)
    sql += " ORDER BY id DESC LIMIT 1"
    with _conn(db_path) as c:
        row = c.execute(sql, args).fetchone()
        return dict(row) if row else None


def settle_reflow(reflow_id: int, retest_score: float, status: str,
                  db_path: str | None = None) -> dict:
    """结算回流任务：写重测分数与终态（completed / given_up）。"""
    with _conn(db_path) as c:
        c.execute(
            "UPDATE reflow_logs SET retest_score=?, status=?, completed_at=? "
            "WHERE id=?",
            (retest_score, status, datetime.now().isoformat(timespec="seconds"),
             reflow_id),
        )
        row = c.execute("SELECT * FROM reflow_logs WHERE id=?", (reflow_id,)).fetchone()
        return dict(row)


def list_reflows(user_id: str, db_path: str | None = None) -> list[dict]:
    """该用户全部回流记录（E 阶段统计 / C 阶段 analyze 用）。"""
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT * FROM reflow_logs WHERE user_id=? ORDER BY id",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def log_llm_call(caller: str, model: str, prompt_tokens: int,
                 completion_tokens: int, total_tokens: int, latency_ms: int,
                 attempts: int = 1, expanded: bool = False,
                 source: str = "api", user_id: str | None = None,
                 session_id: str | None = None,
                 db_path: str | None = None) -> None:
    """记录一次 LLM 调用（可观测性：token 用量 + 耗时，PLAN 8 多 Agent 效率评估）。

    user_id/session_id 由 model 层 hook 自动附带（配额计费 + D4 session trace）。
    """
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO llm_logs (caller, model, prompt_tokens, completion_tokens, "
            "total_tokens, latency_ms, attempts, expanded, source, user_id, "
            "session_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (caller, model, prompt_tokens, completion_tokens, total_tokens,
             latency_ms, attempts, int(expanded), source, user_id, session_id,
             datetime.now().isoformat(timespec="seconds")),
        )


def llm_stats(db_path: str | None = None) -> dict:
    """LLM 调用统计（按环节聚合）：次数 / token / 耗时，供报告与统计使用。"""
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT caller, COUNT(*) calls, SUM(total_tokens) tokens, "
            "SUM(latency_ms) latency_ms, SUM(expanded) expanded "
            "FROM llm_logs GROUP BY caller ORDER BY tokens DESC",
        ).fetchall()
        total = c.execute(
            "SELECT COUNT(*) calls, COALESCE(SUM(total_tokens),0) tokens, "
            "COALESCE(SUM(latency_ms),0) latency_ms FROM llm_logs",
        ).fetchone()
    return {
        "by_caller": [dict(r) for r in rows],
        "total": {"calls": total["calls"], "tokens": total["tokens"],
                  "latency_ms": total["latency_ms"]},
    }


# ==================== F 阶段：审计（L5）与幂等（L2）（PLAN 22） ====================

def audit_log(event: str, user_id: str | None = None, session_id: str | None = None,
              risk: int = 0, detail: dict | str | None = None,
              db_path: str | None = None) -> None:
    """写一条审计日志（安全 L5 全链路审计）。

    只记安全相关事件（注入命中/鉴权失败/幂等命中/管理访问/测评提交），
    常规业务读不记（噪音）。detail 为 dict 时序列化 JSON 存储。
    """
    if isinstance(detail, dict):
        detail = json.dumps(detail, ensure_ascii=False)
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO audit_logs (user_id, session_id, event, risk, detail, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, session_id, event, int(risk), detail,
             datetime.now().isoformat(timespec="seconds")),
        )


def list_audit(user_id: str | None = None, limit: int = 200,
               db_path: str | None = None) -> list[dict]:
    """审计日志查询（最新在前，/api/audit 管理端点数据源）。"""
    sql = "SELECT * FROM audit_logs"
    args: list = []
    if user_id:
        sql += " WHERE user_id=?"
        args.append(user_id)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn(db_path) as c:
        rows = c.execute(sql, args).fetchall()
        return [dict(r) for r in rows]


def idempotency_get(req_key: str, db_path: str | None = None) -> dict | None:
    """按 request_id 取首次响应（安全 L2 幂等：重放返回原响应，不再计分）。"""
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM idempotency_keys WHERE req_key=?",
                        (req_key,)).fetchone()
        return dict(row) if row else None


def idempotency_put(req_key: str, user_id: str, response: str,
                    db_path: str | None = None) -> None:
    """存 request_id → 首次响应。顺带清理超过 TTL 的旧键（量小，每次顺手删）。"""
    now = datetime.now()
    cutoff = (now - timedelta(seconds=config.IDEMPOTENCY_TTL_SECONDS)) \
        .isoformat(timespec="seconds")
    with _conn(db_path) as c:
        c.execute("DELETE FROM idempotency_keys WHERE created_at < ?", (cutoff,))
        c.execute(
            "INSERT OR REPLACE INTO idempotency_keys (req_key, user_id, response, created_at) "
            "VALUES (?,?,?,?)",
            (req_key, user_id, response, now.isoformat(timespec="seconds")),
        )
