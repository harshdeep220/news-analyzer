/**
 * News Analyzer — Chat UI with Gemini-style inline agent trace.
 *
 * The trace renders INSIDE the chat area as a collapsible "Analyzing…"
 * block above the response, exactly like Gemini's "Thinking" section.
 *
 * SSE fix: uses fetch + ReadableStream instead of EventSource to avoid
 * Django buffering issues with the native EventSource API.
 */

// ─── State ───────────────────────────────────────────────────────────────────
let currentSessionId = INITIAL_SESSION_ID || null;
let abortController = null;
let isProcessing = false;

// Pipeline step definitions
const STEPS = [
    { id: "session",         label: "Loading session" },
    { id: "intent_planner",  label: "Analyzing intent" },
    { id: "rag_agent",       label: "Searching knowledge base" },
    { id: "web_search",      label: "Searching the web" },
    { id: "data_validator",  label: "Validating sources" },
    { id: "crag_grader",     label: "Grading relevance" },
    { id: "credibility",     label: "Scoring credibility" },
    { id: "bias",            label: "Detecting bias" },
    { id: "comparator",      label: "Comparing sources" },
    { id: "synthesis",       label: "Synthesizing answer" },
    { id: "hallucination",   label: "Checking claims" },
    { id: "critic",          label: "Quality review" },
    { id: "formatter",       label: "Formatting response" },
];

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const ta = document.getElementById("query-input");

    ta.addEventListener("input", () => {
        ta.style.height = "auto";
        ta.style.height = ta.scrollHeight + "px";
    });

    ta.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });

    if (!currentSessionId) createNewSession();
});

// ─── Submit ──────────────────────────────────────────────────────────────────

function handleSubmit(e) {
    e.preventDefault();
    const input = document.getElementById("query-input");
    const q = input.value.trim();
    if (!q || isProcessing) return;
    submitQuery(q);
    input.value = "";
    input.style.height = "auto";
}

function submitQuery(query) {
    if (isProcessing) return;
    if (!currentSessionId) {
        createNewSession(() => submitQuery(query));
        return;
    }

    isProcessing = true;
    document.getElementById("btn-send").disabled = true;

    const welcome = document.getElementById("welcome-msg");
    if (welcome) welcome.style.display = "none";

    // User message
    addMsg("user", query);

    // Create inline trace block (Gemini-style)
    const traceId = "trace-" + Date.now();
    createTraceBlock(traceId);

    // Start SSE via fetch (avoids Django buffering issue with EventSource)
    const url = `/api/chat?q=${encodeURIComponent(query)}&session_id=${currentSessionId}`;

    abortController = new AbortController();
    const startTime = Date.now();

    fetch(url, { signal: abortController.signal })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        finishProcessing(traceId);
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });

                    // Parse SSE lines
                    const lines = buffer.split("\n");
                    buffer = lines.pop(); // keep incomplete line

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                handleEvent(data, traceId, startTime);
                            } catch (e) { /* skip malformed */ }
                        }
                    }

                    read();
                }).catch(() => finishProcessing(traceId));
            }
            read();
        })
        .catch(() => finishProcessing(traceId));
}


// ─── SSE Event Handler ──────────────────────────────────────────────────────

function handleEvent(data, traceId, startTime) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    if (data.type === "status") {
        markStep(traceId, data.agent, "done", data.content, elapsed);
        // Activate next step
        const idx = STEPS.findIndex(s => s.id === data.agent);
        if (idx >= 0 && idx < STEPS.length - 1) {
            markStep(traceId, STEPS[idx + 1].id, "running");
        }
        updateTraceHeader(traceId, data.agent, elapsed);
    }
    else if (data.type === "done") {
        renderResult(data.result);
        collapseTrace(traceId, elapsed);
        finishProcessing(traceId);
    }
    else if (data.type === "error") {
        addMsg("assistant", `An error occurred: ${data.content || "Pipeline failed"}`);
        collapseTrace(traceId, elapsed);
        finishProcessing(traceId);
    }
}

function finishProcessing(traceId) {
    isProcessing = false;
    document.getElementById("btn-send").disabled = false;
    abortController = null;
    refreshSessionList();
}


// ─── Inline Trace Block (Gemini-style) ──────────────────────────────────────

function createTraceBlock(traceId) {
    const container = document.getElementById("messages");

    const block = document.createElement("div");
    block.className = "msg msg-assistant";
    block.id = traceId;
    block.innerHTML = `
        <div class="trace-block">
            <button class="trace-toggle open" onclick="toggleTrace('${traceId}')">
                <span class="trace-spinner"></span>
                <span class="trace-label">Analyzing...</span>
                <span class="toggle-icon">▶</span>
            </button>
            <div class="trace-body open" id="${traceId}-body">
                ${STEPS.map((s, i) =>
                    `<div class="trace-step waiting" id="${traceId}-${s.id}">
                        <span class="step-icon">${i === 0 ? '' : '○'}</span>
                        <span class="step-name">${s.label}</span>
                        <span class="trace-elapsed"></span>
                    </div>`
                ).join('')}
            </div>
        </div>
    `;

    container.appendChild(block);

    // Mark first step as running
    const first = document.getElementById(`${traceId}-${STEPS[0].id}`);
    if (first) {
        first.className = "trace-step running";
        first.querySelector(".step-icon").innerHTML = "";
    }

    scrollBottom();
}

function markStep(traceId, stepId, status, detail, elapsed) {
    const el = document.getElementById(`${traceId}-${stepId}`);
    if (!el) return;

    el.className = `trace-step ${status}`;
    const icon = el.querySelector(".step-icon");

    if (status === "done") {
        icon.textContent = "✓";
        if (elapsed) {
            el.querySelector(".trace-elapsed").textContent = elapsed + "s";
        }
    } else if (status === "running") {
        icon.innerHTML = "";  // CSS handles spinner via ::after
    }

    scrollBottom();
}

function updateTraceHeader(traceId, agentId, elapsed) {
    const block = document.getElementById(traceId);
    if (!block) return;
    const step = STEPS.find(s => s.id === agentId);
    const label = block.querySelector(".trace-label");
    if (label && step) {
        label.textContent = `${step.label}... (${elapsed}s)`;
    }
}

function collapseTrace(traceId, elapsed) {
    const block = document.getElementById(traceId);
    if (!block) return;

    // Update header to final state
    const label = block.querySelector(".trace-label");
    if (label) label.textContent = `Analysis complete (${elapsed}s)`;

    // Replace spinner with checkmark
    const spinner = block.querySelector(".trace-spinner");
    if (spinner) {
        spinner.className = "";
        spinner.textContent = "✓";
        spinner.style.color = "var(--green)";
        spinner.style.fontWeight = "700";
        spinner.style.fontSize = "14px";
    }

    // Mark all remaining steps as done
    block.querySelectorAll(".trace-step.running, .trace-step.waiting").forEach(el => {
        el.className = "trace-step done";
        el.querySelector(".step-icon").textContent = "✓";
    });

    // Collapse
    const toggle = block.querySelector(".trace-toggle");
    const body = document.getElementById(`${traceId}-body`);
    if (toggle) toggle.classList.remove("open");
    if (body) body.classList.remove("open");
}

function toggleTrace(traceId) {
    const toggle = document.querySelector(`#${traceId} .trace-toggle`);
    const body = document.getElementById(`${traceId}-body`);
    if (toggle && body) {
        toggle.classList.toggle("open");
        body.classList.toggle("open");
    }
}


// ─── Message Rendering ──────────────────────────────────────────────────────

function addMsg(role, content) {
    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `msg msg-${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "user") {
        bubble.textContent = content;
    } else {
        bubble.innerHTML = md(content);
    }

    div.appendChild(bubble);
    container.appendChild(div);
    scrollBottom();
}

function renderResult(result) {
    if (!result) return;

    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "msg msg-assistant";

    let html = '<div class="bubble">';
    const answer = result.answer_html || result.answer || "No response generated.";
    html += md(answer);

    // Credibility badges
    if (result.credibility && Object.keys(result.credibility).length) {
        html += '<div class="intel"><div class="intel-label">Source Credibility</div>';
        for (const [url, cred] of Object.entries(result.credibility)) {
            const s = cred.total;
            const cls = s > 70 ? "badge-green" : s > 40 ? "badge-amber" : "badge-red";
            let domain;
            try { domain = new URL(url).hostname.replace("www.", ""); } catch { domain = url.slice(0, 30); }
            html += `<span class="badge ${cls}" title='${esc(JSON.stringify(cred.signals || {}))}'>
                <span class="badge-score">${s}</span> ${esc(domain)}
            </span>`;
        }
        html += '</div>';
    }

    // Bias
    if (result.bias) {
        const visible = Object.entries(result.bias).filter(([_, b]) => b.lean && b.lean !== "insufficient_content");
        if (visible.length) {
            html += '<div class="intel"><div class="intel-label">Bias Analysis</div>';
            for (const [url, bias] of visible) {
                let domain;
                try { domain = new URL(url).hostname.replace("www.", ""); } catch { domain = url.slice(0, 30); }
                html += `<span class="badge badge-amber">${esc(bias.lean)} (${(bias.confidence * 100).toFixed(0)}%) — ${esc(domain)}</span>`;
            }
            html += '</div>';
        }
    }

    // Hallucination
    if (result.hallucination?.ungrounded_claims?.length) {
        const score = (result.hallucination.grounding_score * 100).toFixed(0);
        html += `<div class="halluc-box"><div class="halluc-title">Grounding: ${score}% — ${result.hallucination.ungrounded_claims.length} unverified claim(s)</div>`;
        result.hallucination.ungrounded_claims.slice(0, 5).forEach(c => {
            html += `<div class="halluc-claim">${esc(c)}</div>`;
        });
        html += '</div>';
    }

    html += '</div>';
    div.innerHTML = html;
    container.appendChild(div);
    scrollBottom();
}


// ─── Session Management ─────────────────────────────────────────────────────

function createNewSession(cb) {
    fetch("/api/session/new", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            currentSessionId = data.session_id;
            refreshSessionList();
            document.getElementById("messages").innerHTML = "";
            const w = document.getElementById("welcome-msg");
            if (w) w.style.display = "flex";
            if (cb) cb();
        })
        .catch(e => console.error("Session create failed:", e));
}

function switchSession(id) {
    if (id === currentSessionId) return;
    if (abortController) { abortController.abort(); abortController = null; }
    isProcessing = false;
    document.getElementById("btn-send").disabled = false;
    currentSessionId = id;

    document.querySelectorAll(".session-item").forEach(el =>
        el.classList.toggle("active", el.dataset.sessionId === id)
    );

    loadHistory(id);
}

function loadHistory(id) {
    const container = document.getElementById("messages");
    container.innerHTML = "";

    fetch(`/api/session/${id}/history`)
        .then(r => r.json())
        .then(data => {
            const w = document.getElementById("welcome-msg");
            if (data.messages?.length) {
                if (w) w.style.display = "none";
                data.messages.forEach(m => { if (m.role !== "system") addMsg(m.role, m.content); });
            } else {
                if (w) w.style.display = "flex";
            }
        });
}

function refreshSessionList() {
    fetch("/api/sessions")
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("session-list");
            if (!data.sessions?.length) {
                list.innerHTML = '<div class="session-empty">No sessions yet</div>';
                return;
            }
            list.innerHTML = data.sessions.map(s =>
                `<div class="session-item ${s.id === currentSessionId ? "active" : ""}"
                     data-session-id="${s.id}"
                     onclick="switchSession('${s.id}')">
                    <div class="session-label">${esc(s.topic_label)}</div>
                    <div class="session-meta">${s.message_count} messages</div>
                </div>`
            ).join("");
        });
}

document.getElementById("btn-new-session").addEventListener("click", () => createNewSession());

// ─── Utilities ──────────────────────────────────────────────────────────────

function scrollBottom() {
    const el = document.getElementById("chat-scroll");
    el.scrollTop = el.scrollHeight;
}

function esc(t) {
    const m = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return String(t).replace(/[&<>"']/g, c => m[c]);
}

function md(text) {
    if (!text) return "";
    let h = esc(text);
    h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    h = h.replace(/\*(.+?)\*/g, "<em>$1</em>");
    h = h.replace(/`(.+?)`/g, "<code>$1</code>");
    h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    h = h.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    h = h.replace(/^\* (.+)$/gm, "<li>$1</li>");
    h = h.replace(/^- (.+)$/gm, "<li>$1</li>");
    h = h.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
    h = h.replace(/\n\n/g, "</p><p>");
    h = `<p>${h}</p>`;
    h = h.replace(/<p>\s*<\/p>/g, "");
    h = h.replace(/\n/g, "<br>");
    return h;
}
