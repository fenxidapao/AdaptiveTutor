/* FeynmanTutor 前端逻辑：无框架原生 JS，步骤式学习闭环 */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const state = {
    user: "u0",
    session: localStorage.getItem("ft_session") || null,
    group: null, // C2 自变量（P0-1 修复）：feynman=先答后讲+追问；lecture=直接标准讲解
    course: "python",
    kpList: [],
    // 费曼会话状态（前端保存 transcript，后端无状态）
    feynman: { kpId: null, transcript: [], round: 0, maxRounds: 3 },
    // 练习状态
    practice: { exs: [], idx: 0 },
    chart: null,
    reportChart: null,
  };

  /* ---------------- 通用 ---------------- */
  async function api(path, opts = {}) {
    const r = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status}: ${t.slice(0, 120)}`);
    }
    return r.json();
  }

  // C1：登录态携带（GET 加 query / POST 加 body 字段 session_id）
  function sessUrl(base) {
    // 自动判断分隔符：base 已有 query 用 &，否则用 ?（避免拼出 /user&session_id 的 bug）
    return state.session
      ? `${base}${base.includes("?") ? "&" : "?"}session_id=${encodeURIComponent(state.session)}`
      : base;
  }
  function sessBody(b = {}) {
    if (state.session) b.session_id = state.session;
    return b;
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function switchStep(name) {
    $$(".steps button").forEach(b => b.classList.toggle("active", b.dataset.step === name));
    $$(".view").forEach(v => v.classList.toggle("hidden", v.id !== `view-${name}`));
  }

  async function loadKps() {
    const d = await api(`/api/learning-pack/${state.course}`);
    state.kpList = d.knowledge_points;
    $("#kpSelect").innerHTML = state.kpList
      .map(k => `<option value="${k.kp_id}">[${k.chapter}] ${esc(k.title)}</option>`).join("");
  }

  /* ---------------- 登录 / 注册（C1，PLAN 18.3） ---------------- */
  async function doAuth(path) {
    const uid = $("#userId").value.trim();
    const pwd = $("#userPwd").value;
    if (!uid || !pwd) { alert("请输入账号和密码"); return; }
    try {
      const r = await api(path, { method: "POST", body: JSON.stringify({ user_id: uid, password: pwd }) });
      state.user = r.user.user_id;
      state.session = r.session_id;
      state.group = r.user.group_name || null;
      localStorage.setItem("ft_session", r.session_id);
      $("#userPwd").value = "";
      const g = r.user.group_name ? `（实验组: ${r.user.group_name}）` : "";
      $("#loginHint").textContent = `已登录 ${state.user}${g}`;
      loadKps(); loadQuiz("pretest", "#pretestBox"); loadQueue();
    } catch (e) { alert("失败：" + e.message); }
  }

  function doLogout() {
    const sid = state.session;
    state.session = null; state.user = "u0"; state.group = null;
    localStorage.removeItem("ft_session");
    $("#loginHint").textContent = "未登录（演示模式 u0）";
    if (sid) api("/api/logout", { method: "POST", body: JSON.stringify({ session_id: sid }) }).catch(() => {});
  }

  /* ---------------- 前测 / 后测 ---------------- */
  async function loadQuiz(kind, boxSel) {
    const box = $(boxSel);
    box.innerHTML = "<p class='hint'>加载中…</p>";
    state.quizStart = Date.now();  // C1：计时作弊检测
    const qs = await api(sessUrl(`/api/quiz/${state.course}/${kind}?user_id=${state.user}`));
    if (!qs.length) { box.innerHTML = "<p class='hint'>无题目</p>"; return; }
    box.innerHTML = "";
    qs.forEach((q, i) => {
      const div = document.createElement("div");
      div.className = "q";
      div.dataset.ex = q.ex_id;
      let body;
      if (q.type === "mcq") {
        body = `<div class="q-options">${q.options.map((o, j) =>
          `<label><input type="radio" name="q${i}" value="${j}">${esc(o)}</label>`).join("")}</div>`;
      } else {
        body = `<textarea class="q-input" placeholder="输入 Python 代码…" rows="3"></textarea>`;
      }
      div.innerHTML = `<div class="q-title">${i + 1}. ${esc(q.prompt)}</div>${body}
        <div class="q-feedback"></div>`;
      box.appendChild(div);
    });
    const btn = document.createElement("button");
    btn.className = "primary mt";
    btn.textContent = `提交${kind === "pretest" ? "前测" : "后测"}`;
    btn.onclick = () => submitQuiz(kind, box);
    box.appendChild(btn);
  }

  async function submitQuiz(kind, box) {
    const answers = {};
    $$(".q", box).forEach(div => {
      const ex = div.dataset.ex;
      const radio = $('input[type=radio]:checked', div);
      if (radio) answers[ex] = radio.value;
      else { const ta = $("textarea", div); if (ta?.value.trim()) answers[ex] = ta.value.trim(); }
    });
    const btn = $("button.primary", box);
    btn.disabled = true; btn.textContent = "判题中…";
    try {
      // C1：答题耗时（作弊检测依据，EXPERIMENT.md）
      const elapsed = state.quizStart ? Math.round((Date.now() - state.quizStart) / 1000) : null;
      const res = await api(`/api/quiz/${state.course}/${kind}/submit`, {
        method: "POST",
        // C1-3：mode 由后端按实验组强制，前端不再指定（防篡改分组）
        body: JSON.stringify(sessBody({ user_id: state.user, answers, elapsed_seconds: elapsed })),
      });
      // 每题标注对错
      res.details.forEach(d => {
        const div = $$(".q", box).find(d => d.dataset.ex === d.ex_id);
        const fb = div ? $(".q-feedback", div) : null;
        if (fb) {
          fb.textContent = (d.correct ? "✓ " : "✗ ") + d.feedback;
          fb.className = "q-feedback " + (d.correct ? "ok" : "err");
        }
      });
      const fb = document.createElement("div");
      fb.className = "feedback " + (res.correct / res.total >= 0.6 ? "ok" : "err");
      fb.textContent = `${kind === "pretest" ? "前测" : "后测"}得分：${res.correct}/${res.total}（${Math.round(res.score * 100)}%）`;
      box.appendChild(fb);
      // E1 学习回流（PLAN 20）：后测提交后若触发回流，就地提示
      if (kind === "posttest" && res.reflow && res.reflow.triggered) {
        const rf = document.createElement("div");
        if (res.reflow.passed) {
          rf.className = "feedback ok";
          rf.textContent = `重测达标 ${Math.round(res.reflow.reflow.retest_score * 100)}%（≥${Math.round(res.reflow.pass_score * 100)}%）— 学习闭环完成 ✅`;
        } else if (res.reflow.gave_up) {
          rf.className = "feedback err";
          rf.textContent = `已达回流上限 ${res.reflow.max_rounds} 轮仍未达标，建议带着错题求助后再来。`;
        } else if (res.reflow.passed === null) {
          rf.className = "feedback err";
          rf.textContent = `后测未达标 → 已生成第 ${res.reflow.round} 轮回流任务，去「报告」页继续学习薄弱点。`;
        } else {
          rf.className = "feedback err";
          rf.textContent = `重测未达标 → 进入第 ${res.reflow.round} 轮回流，去「报告」页重新学习。`;
        }
        box.appendChild(rf);
      }
      btn.remove();
      markStepDone(kind === "pretest" ? "pretest" : "posttest");
    } catch (e) {
      alert("提交失败：" + e.message);
      btn.disabled = false; btn.textContent = "重试";
    }
  }

  /* ---------------- 诊断 ---------------- */
  function weakId(w) {
    return (w && typeof w === "object") ? w.kp_id : w;
  }

  async function runDiagnose() {
    const btn = $("#btnDiagnose");
    btn.disabled = true; btn.textContent = "诊断中…";
    try {
      const p = await api(`/api/diagnose/${state.user}${state.session ? "?session_id=" + encodeURIComponent(state.session) : ""}`, { method: "POST" });
      const weak = (p.weak_points || []).map(w => {
        const id = weakId(w);
        const kp = state.kpList.find(k => k.kp_id === id);
        const reason = (w && w.reason) ? ` <span class="hint">— ${esc(w.reason)}</span>` : "";
        const ev = (w && w.evidence && w.evidence.length)
          ? `<span class="hint">（证据: ${esc(w.evidence.slice(0, 3).join(", "))}）</span>` : "";
        return `<span class="tag weak">${esc(kp ? kp.title : id)}${reason}${ev}</span>`;
      }).join("");
      $("#diagnoseBox").innerHTML = `
        <table class="stats">
          <tr><td>薄弱知识点</td><td>${weak || "<span class='hint'>暂无</span>"}</td></tr>
          <tr><td>平均正确率</td><td><span class="big-num">${Math.round((p.avg_correct || 0) * 100)}%</span></td></tr>
          <tr><td>学习偏好</td><td>${esc(p.learning_style || "简答")}</td></tr>
        </table>`;
      markStepDone("diagnose");
    } catch (e) {
      alert("诊断失败：" + e.message);
    }
    btn.disabled = false; btn.textContent = "重新诊断";
  }

  /* ---------------- 费曼学习 ---------------- */
  function renderFeynman() {
    const box = $("#learnBox");
    const f = state.feyman;
    box.innerHTML = "";
    // 对话区
    const chat = document.createElement("div");
    chat.className = "chat";
    chat.id = "feynmanChat";
    f.transcript.forEach(t => {
      const m = document.createElement("div");
      m.className = "msg " + (t.role === "user" ? "user" : "coach");
      m.textContent = t.content;
      chat.appendChild(m);
    });
    box.appendChild(chat);
    // 输入区
    const row = document.createElement("div");
    row.className = "row mt";
    row.innerHTML = `<textarea id="feynmanInput" rows="2"
      placeholder="${f.transcript.length ? "回答教练的追问…" : "用你自己的话讲解这个知识点（可举例）…"}"></textarea>
      <button id="feynmanSend" class="primary">${f.transcript.length ? "回答" : "开始讲"}</button>`;
    box.appendChild(row);

    $("#feynmanSend").onclick = async () => {
      const input = $("#feynmanInput");
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      f.transcript.push({ role: "user", content: text });
      renderFeynman(); // 先显示学生的话
      const sendBtn = $("#feynmanSend");
      sendBtn.disabled = true; sendBtn.textContent = "思考中…";

      if (f.round >= f.maxRounds - 1) {
        // 3 轮已满 → 总结盲点
        try {
          const r = await api("/api/feynman/summarize", {
            method: "POST",
            body: JSON.stringify(sessBody({ course: state.course, kp_id: f.kpId, transcript: f.transcript, user_id: state.user })),
          });
          const g = document.createElement("div");
          g.className = "msg gap";
          g.innerHTML = `<b>发现的盲点：</b><br>${(r.gaps || []).map(x => "• " + esc(x)).join("<br>")}`;
          $("#feynmanChat").appendChild(g);
          showExplain();
        } catch (e) { alert("总结失败：" + e.message); }
      } else {
        try {
          const r = await api("/api/feynman/turn", {
            method: "POST",
            body: JSON.stringify(sessBody({ course: state.course, kp_id: f.kpId, transcript: f.transcript, user_id: state.user })),
          });
          f.transcript.push({ role: "assistant", content: r.coach });
          f.round += 1;
          renderFeynman();
        } catch (e) { alert("追问失败：" + e.message); }
      }
    };
    $("#feynmanInput").focus();
  }

  async function showExplain() {
    const box = $("#learnBox");
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = "<h3>标准讲解</h3><div id='explainText' class='hint'>加载中…</div>";
    box.appendChild(card);
    try {
      const r = await api(sessUrl(`/api/explain/${state.course}/${state.feyman.kpId}?user_id=${state.user}`));
      $("#explainText").textContent = r.explanation;
    } catch (e) {
      $("#explainText").textContent = `讲解加载失败：${e.message}`;
      return;
    }
    // 接着练习
    const pBtn = document.createElement("button");
    pBtn.className = "primary mt";
    pBtn.textContent = "开始练习";
    pBtn.onclick = () => startPractice();
    box.appendChild(pBtn);
  }

  async function startPractice() {
    const box = $("#learnBox");
    const exs = await api(`/api/exercises/${state.course}?kp_id=${state.feyman.kpId}`);
    if (!exs.length) { box.innerHTML += "<p class='hint'>该知识点暂无练习题</p>"; return; }
    state.practice = { exs, idx: 0 };
    box.innerHTML += "<h3 class='mt'>练习（沙箱判题）</h3><div id='practiceBox'></div>";
    renderPractice();
  }

  function renderPractice() {
    const p = state.practice;
    if (p.idx >= p.exs.length) {
      $("#practiceBox").innerHTML = "<div class='feedback ok'>练习完成！可进入后测。</div>";
      markStepDone("learn");
      return;
    }
    const ex = p.exs[p.idx];
    $("#practiceBox").innerHTML = `
      <div class="q">
        <div class="q-title">练习 ${p.idx + 1}. ${esc(ex.prompt)}</div>
        ${ex.type === "mcq"
          ? `<div class="q-options">${ex.options.map((o, j) =>
              `<label><input type="radio" name="p" value="${j}">${esc(o)}</label>`).join("")}</div>`
          : `<textarea class="q-input" rows="3" placeholder="输入 Python 代码…"></textarea>`}
        <div class="q-feedback"></div>
        <button class="primary mt" id="practiceSubmit">提交</button>
      </div>`;
    $("#practiceSubmit").onclick = async () => {
      let answer;
      if (ex.type === "mcq") {
        const r = $('input[name=p]:checked');
        if (!r) { alert("请选择答案"); return; }
        answer = r.value;
      } else {
        answer = $(".q-input").value.trim();
        if (!answer) { alert("请输入代码"); return; }
      }
      const btn = $("#practiceSubmit");
      btn.disabled = true; btn.textContent = "判题中…";
      try {
        // 幂等键（安全 L2）：每次点击唯一，网络重试/双击重放时服务端返回
        // 首次响应、不重复计分（连续故意提交同一答案不受影响——新点击有新键）
        const res = await api("/api/grade", {
          method: "POST",
          body: JSON.stringify(sessBody({ user_id: state.user, ex_id: ex.ex_id, answer,
                                          request_id: (crypto.randomUUID ? crypto.randomUUID()
                                                                   : String(Date.now()) + Math.random()) })),
        });
        const fb = $(".q-feedback");
        fb.textContent = (res.correct ? "✓ " : "✗ ") + res.feedback;
        fb.className = "q-feedback " + (res.correct ? "ok" : "err");
        if (!res.correct) {
          // E2 练习策略切换（PLAN 20.3）：连续失败 → hint → 讲解 → 前置复习
          if (res.explanation) fb.textContent += "\n📖 考点: " + res.explanation;
          if (res.strategy === "explain") {
            fb.textContent += "\n📚 连续失败 2 次：先看标准讲解和对比举例，再回来做题。";
          } else if (res.strategy === "prereq") {
            fb.textContent += "\n📚 连续多次失败：先复习前置知识点——" +
              (res.prereq_titles || []).join("、");
          }
        }
        if (res.correct) {
          btn.textContent = "下一题 →";
          btn.onclick = () => { p.idx += 1; renderPractice(); };
          btn.disabled = false;
        } else {
          btn.disabled = false; btn.textContent = "再试一次（最多 3 次）";
        }
      } catch (e) { alert("判题失败：" + e.message); btn.disabled = false; }
    };
  }

  /* ---------------- 报告 ---------------- */
  async function loadReport() {
    const box = $("#reportBox");
    box.innerHTML = "<p class='hint'>加载中…</p>";
    try {
      const r = await api(sessUrl(`/api/report/${state.course}/${state.user}`));
      if (r.pre == null && r.post == null) {
        box.innerHTML = "<p class='hint'>还没有前后测数据，先跑前测和后测。</p>";
        $("#reportChartBox").style.display = "none";
        return;
      }
      $("#reportChartBox").style.display = "";
      const gain = r.gain_pp == null ? "—" : `${r.gain_pp > 0 ? "+" : ""}${r.gain_pp}pp`;
      const weak = (r.weak_points || []).map(w => {
        const id = weakId(w);
        const kp = state.kpList.find(k => k.kp_id === id);
        return `<span class="tag weak">${esc(kp ? kp.title : id)}</span>`;
      }).join("");
      box.innerHTML = `
        <table class="stats">
          <tr><td>前测正确率</td><td><span class="big-num">${r.pre == null ? "—" : Math.round(r.pre * 100) + "%"}</span></td></tr>
          <tr><td>后测正确率</td><td><span class="big-num">${r.post == null ? "—" : Math.round(r.post * 100) + "%"}</span></td></tr>
          <tr><td>提升</td><td><span class="big-num">${gain}</span></td></tr>
          <tr><td>薄弱点</td><td>${weak || "<span class='hint'>暂无</span>"}</td></tr>
        </table>`;
      renderReportChart(r.by_chapter || []);
      // E1 回流卡片（PLAN 20.2）：active=有回流任务在身 → 引导继续学习薄弱点
      const rf = await api(sessUrl(`/api/reflow/${state.course}/${state.user}`)).catch(() => null);
      renderReflowCard(rf);
    } catch (e) {
      box.innerHTML = `<div class='feedback err'>报告加载失败：${esc(e.message)}</div>`;
      $("#reportChartBox").style.display = "none";
    }
  }

  /* E1 回流卡片（"继续学习"引导） */
  function renderReflowCard(rf) {
    const box = $("#reflowBox");
    if (!box) return;
    if (!rf) { box.classList.add("hidden"); return; }
    if (rf.active) {
      const weak = (rf.weak_kps || []).map(id => {
        const kp = state.kpList.find(k => k.kp_id === id);
        return `<span class="tag weak">${esc(kp ? kp.title : id)}</span>`;
      }).join("");
      box.innerHTML = `
        <h3>学习回流（第 ${rf.round}/${rf.max_rounds} 轮）<span class="hint">后测未达标，重新学薄弱点</span></h3>
        <p>薄弱点：${weak || "<span class='hint'>无</span>"}</p>
        <p class="hint">重测后测达标线 ${Math.round(rf.pass_score * 100)}% · 不达标自动续轮，最多 ${rf.max_rounds} 轮</p>
        <button class="primary mt" id="btnReflow">继续学习薄弱点 →</button>`;
      box.classList.remove("hidden");
      $("#btnReflow").onclick = () => {
        const first = (rf.weak_kps || [])[0];
        if (first && state.kpList.some(k => k.kp_id === first)) $("#kpSelect").value = first;
        switchStep("learn");
        $("#btnStartKp").click();
      };
      return;
    }
    const label = ({ completed: "✅ 回流完成，闭环达标", given_up: "⏹ 已达回流上限，本轮结束" })[rf.last_status];
    if (label) {
      box.innerHTML = `<h3>学习回流 <span class="hint">${label}</span></h3>`;
      box.classList.remove("hidden");
    } else {
      box.classList.add("hidden");
    }
  }

  /* E3 今日学习队列（PLAN 20.4）：到期复习 + 未掌握知识点 */
  async function loadQueue() {
    const box = $("#queueBox");
    if (!box) return;
    try {
      const d = await api(sessUrl(`/api/queue/${state.course}/${state.user}`));
      const q = d.queue || [];
      if (!q.length) { box.classList.add("hidden"); return; }
      box.innerHTML = `<h3>今日学习队列 <span class="hint">系统按掌握度排的学习顺序</span></h3>
        <ol class="queue">${q.map(item => `
          <li>
            <span class="q-reason ${item.reason}">${item.reason === "review" ? "复习" : "补弱"}</span>
            ${esc(item.title)}
            <span class="hint">${item.reason === "review" ? `到期 ${item.next_review}` : `掌握度 ${Math.round(item.mastery * 100)}%`}</span>
            ${item.reason === "weak" ? `<div class="q-bar"><i style="width:${Math.round(item.mastery * 100)}%"></i></div>` : ""}
          </li>`).join("")}</ol>`;
      box.classList.remove("hidden");
    } catch (e) { box.classList.add("hidden"); }
  }

  function renderReportChart(rows) {
    const cv = $("#reportChart");
    if (!cv) return;
    if (state.reportChart) state.reportChart.destroy();
    // Web 实验流只有 'all' 一个聚合章节，图例显示"总体"
    const chapters = rows.map(r => r.chapter === "all" ? "总体" : r.chapter);
    const pre = rows.map(r => r.pre == null ? null : Math.round(r.pre * 100));
    const post = rows.map(r => r.post == null ? null : Math.round(r.post * 100));
    if (!chapters.length) { $("#reportChartBox").style.display = "none"; return; }
    state.reportChart = new Chart(cv, {
      type: "bar",
      data: {
        labels: chapters,
        datasets: [
          { label: "前测", data: pre, backgroundColor: "rgba(148,163,184,.75)", borderRadius: 6 },
          { label: "后测", data: post, backgroundColor: "rgba(22,163,74,.85)", borderRadius: 6 },
        ],
      },
      options: {
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: c => `${c.dataset.label} ${c.raw}%` } },
        },
        scales: {
          y: { min: 0, max: 100, ticks: { callback: v => v + "%" } },
        },
      },
    });
  }

  /* ---------------- 热力图 ---------------- */
  async function loadHeatmap() {
    const d = await api(sessUrl(`/api/heatmap/${state.user}`));
    const labels = d.cells.map(c => c.title);
    const data = d.cells.map(c => c.mastery);
    const colors = d.cells.map(c => {
      if (c.mastery >= 0.8) return "rgba(22,163,74,.85)";
      if (c.mastery >= 0.5) return "rgba(37,99,235,.75)";
      if (c.mastery > 0) return "rgba(217,119,6,.75)";
      return "rgba(148,163,184,.4)";
    });
    if (state.chart) state.chart.destroy();
    state.chart = new Chart($("#heatmapChart"), {
      type: "bar",
      data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 6 }] },
      options: {
        indexAxis: "y",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => `掌握度 ${Math.round(c.raw * 100)}%` } },
        },
        scales: { x: { max: 1, ticks: { callback: v => Math.round(v * 100) + "%" } } },
      },
    });
    $("#heatmapLegend").innerHTML =
      "🟢 ≥80% 已掌握 · 🔵 ≥50% 学习中 · 🟠 >0 薄弱 · ⚪ 未开始";
  }

  /* ---------------- 步骤导航 ---------------- */
  function markStepDone(step) {
    const btn = $(`.steps button[data-step=${step}]`);
    if (btn && !btn.classList.contains("active")) btn.classList.add("done");
  }

  function bindSteps() {
    $$(".steps button").forEach(b => {
      b.onclick = () => {
        switchStep(b.dataset.step);
        if (b.dataset.step === "posttest") loadQuiz("posttest", "#posttestBox");
        if (b.dataset.step === "report") loadReport();
        if (b.dataset.step === "heatmap") loadHeatmap();
      };
    });
  }

  /* ---------------- 初始化 ---------------- */
  async function init() {
    bindSteps();
    $("#btnLogin").onclick = () => doAuth("/api/login");
    $("#btnRegister").onclick = () => doAuth("/api/register");
    $("#btnLogout").onclick = doLogout;
    if (state.session) {
      // 刷新恢复登录态（服务端校验）
      try {
        const me = await api(`/api/me?session_id=${encodeURIComponent(state.session)}`);
        state.user = me.user_id;
        state.group = me.group_name || null;
        $("#loginHint").textContent = `已登录 ${me.user_id}${me.group_name ? `（实验组: ${me.group_name}）` : ""}`;
        $("#btnLogout").classList.remove("hidden");
      } catch (e) {
        state.session = null; localStorage.removeItem("ft_session");
        $("#loginHint").textContent = "会话已过期，请重新登录";
      }
    } else {
      $("#loginHint").textContent = "演示模式 u0（同学请先注册）";
    }
    $("#btnDiagnose").onclick = runDiagnose;
    $("#btnStartKp").onclick = () => {
      state.feyman = { kpId: $("#kpSelect").value, transcript: [], round: 0, maxRounds: 3 };
      $("#learnBox").innerHTML = "";
      // C2 自变量分支（P0-1 修复）：lecture 组直接看标准讲解，不走费曼追问。
      // 硬约束在服务端 /api/feynman/*（lecture 调用即 403），前端分支只决定 UX。
      if (state.group === "lecture") showExplain();
      else renderFeynman();
    };
    await loadKps();
    loadQueue();
    loadQuiz("pretest", "#pretestBox");
  }

  init();
})();
