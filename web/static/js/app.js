/**
 * NewsForge — Chat UI with Live Agent Trace
 *
 * Uses native EventSource (SSE) — no JS dependencies.
 * Each pipeline event updates the trace panel in real-time:
 *   ○ waiting → ⏳ running → ✅ done
 */

// ─── State ───────────────────────────────────────────────────────────────────
let currentSessionId = INITIAL_SESSION_ID || null;
let activeSource = null;   // EventSource instance
let traceTimer = null;     // Timer interval
let traceStartTime = 0;
let isProcessing = false;

// ─── Agent pipeline steps (in execution order) ──────────────────────────────
const PIPELINE_STEPS = [
    { id: "session",        label: "Session" },
    { id: "intent_planner", label: "Intent" },
    { id: "web_search",     label: "Web Search" },
    { id: "rag_agent",      label: "RAG" },
    { id: "data_validator",  label: "Validate" },
    { id: "crag_grader",    label: "CRAG" },
    { id: "credibility",    label: "Credibility" },
    { id: "bias",           label: "Bias" },
    { id: "comparator",     label: "Compare" },
    { id: "synthesis",      label: "Synthesis" },
    { id: "hallucination",  label: "Hallucination" },
    { id: "critic",         label: "Critic" },
    { id: "formatter",      label: "Format" },
];

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const textarea = document.getElementById("query-input");

    // Auto-resize textarea
    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });

    // Shift+Enter for newline, Enter to submit
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });

    // Create initial session if none exists
    if (!currentSessionId) {
        createNewSession();
    }
});


// ─── Submit Handler ──────────────────────────────────────────────────────────

function handleSubmit(e) {
    e.preventDefault();
    const input = document.getElementById("query-input");
    const query = input.value.trim();
    if (!query || isProcessing) return;

    submitQuery(query);
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

    // Hide welcome, show messages
    const welcome = document.getElementById("welcome-msg");
    if (welcome) welcome.style.display = "none";

    // Add user message
    appendMessage("user", query);

    // Show trace panel
    showTracePanel();

    // Start SSE
    const url = `/api/chat?q=${encodeURIComponent(query)}&session_id=${currentSessionId}`;

    if (activeSource) {
        activeSource.close();
    }

    activeSource = new EventSource(url);

    activeSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleSSEEvent(data);
        } catch (err) {
            console.error("SSE parse error:", err);
        }
    };

    activeSource.onerror = () => {
        activeSource.close();
        activeSource = null;
        stopTrace();
        isProcessing = false;
        document.getElementById("btn-send").disabled = false;
    };
}


// ─── SSE Event Handler ──────────────────────────────────────────────────────

function handleSSEEvent(data) {
    if (data.type === "status") {
        updateTraceStep(data.agent, "done", data.content);
        // Mark next step as running
        activateNextStep(data.agent);
    } else if (data.type === "done") {
        stopTrace();
        renderResult(data.result);
        isProcessing = false;
        document.getElementById("btn-send").disabled = false;
        if (activeSource) {
            activeSource.close();
            activeSource = null;
        }
        refreshSessionList();
    } else if (data.type === "error") {
        stopTrace();
        appendMessage("assistant", `Error: ${data.content || "Pipeline failed"}`);
        isProcessing = false;
        document.getElementById("btn-send").disabled = false;
        if (activeSource) {
            activeSource.close();
            activeSource = null;
        }
    }
}


// ─── Trace Panel ─────────────────────────────────────────────────────────────

function showTracePanel() {
    const panel = document.getElementById("trace-panel");
    const steps = document.getElementById("trace-steps");
    panel.style.display = "block";

    // Build step chips
    steps.innerHTML = PIPELINE_STEPS.map(step =>
        `<div class="trace-step waiting" id="trace-${step.id}">
            <span class="step-icon">○</span>
            <span>${step.label}</span>
        </div>`
    ).join("");

    // Mark first step as running
    const first = document.getElementById("trace-session");
    if (first) {
        first.classList.remove("waiting");
        first.classList.add("running");
        first.querySelector(".step-icon").textContent = "⏳";
    }

    // Start timer
    traceStartTime = Date.now();
    traceTimer = setInterval(() => {
        const elapsed = ((Date.now() - traceStartTime) / 1000).toFixed(1);
        document.getElementById("trace-timer").textContent = elapsed + "s";
    }, 100);
}

function updateTraceStep(agentId, status, content) {
    const step = document.getElementById(`trace-${agentId}`);
    if (!step) return;

    step.className = `trace-step ${status}`;
    if (status === "done") {
        step.querySelector(".step-icon").textContent = "✅";
    } else if (status === "running") {
        step.querySelector(".step-icon").textContent = "⏳";
    }
}

function activateNextStep(currentAgentId) {
    const idx = PIPELINE_STEPS.findIndex(s => s.id === currentAgentId);
    if (idx >= 0 && idx < PIPELINE_STEPS.length - 1) {
        const next = PIPELINE_STEPS[idx + 1];
        const nextEl = document.getElementById(`trace-${next.id}`);
        if (nextEl && nextEl.classList.contains("waiting")) {
            nextEl.classList.remove("waiting");
            nextEl.classList.add("running");
            nextEl.querySelector(".step-icon").textContent = "⏳";
        }
    }
}

function stopTrace() {
    if (traceTimer) {
        clearInterval(traceTimer);
        traceTimer = null;
    }
    // Mark all remaining running steps as done
    document.querySelectorAll(".trace-step.running").forEach(el => {
        el.classList.remove("running");
        el.classList.add("done");
        el.querySelector(".step-icon").textContent = "✅";
    });
}

function hideTracePanel() {
    document.getElementById("trace-panel").style.display = "none";
}


// ─── Message Rendering ──────────────────────────────────────────────────────

function appendMessage(role, content) {
    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = `message message-${role}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (role === "user") {
        contentDiv.textContent = content;
    } else {
        contentDiv.innerHTML = formatMarkdown(content);
    }

    div.appendChild(contentDiv);
    container.appendChild(div);
    scrollToBottom();
}

function renderResult(result) {
    if (!result) return;

    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message message-assistant";

    let html = '<div class="message-content">';

    // Main answer
    const answer = result.answer_html || result.answer || "No response generated";
    html += formatMarkdown(answer);

    // Credibility badges
    if (result.credibility && Object.keys(result.credibility).length > 0) {
        html += '<div class="intel-section">';
        html += '<div class="intel-title">Source Credibility</div>';
        for (const [url, cred] of Object.entries(result.credibility)) {
            const score = cred.total;
            const colorClass = score > 70 ? "cred-high" : score > 40 ? "cred-mid" : "cred-low";
            const domain = new URL(url).hostname.replace("www.", "");
            html += `<span class="cred-badge ${colorClass}" title="${JSON.stringify(cred.signals || {})}">
                <span class="cred-score">${score}</span> ${domain}
            </span>`;
        }
        html += '</div>';
    }

    // Bias indicators
    if (result.bias && Object.keys(result.bias).length > 0) {
        const visibleBias = Object.entries(result.bias).filter(
            ([_, b]) => b.lean && b.lean !== "insufficient_content"
        );
        if (visibleBias.length > 0) {
            html += '<div class="intel-section">';
            html += '<div class="intel-title">Bias Analysis</div>';
            for (const [url, bias] of visibleBias) {
                const domain = new URL(url).hostname.replace("www.", "");
                html += `<span class="cred-badge cred-mid" title="Automated classification">
                    ${bias.lean} (${(bias.confidence * 100).toFixed(0)}%) — ${domain}
                </span>`;
            }
            html += '</div>';
        }
    }

    // Hallucination warnings
    if (result.hallucination && result.hallucination.ungrounded_claims &&
        result.hallucination.ungrounded_claims.length > 0) {
        const score = (result.hallucination.grounding_score * 100).toFixed(0);
        html += `<div class="halluc-warning">
            <div class="halluc-title">Grounding Score: ${score}% — ${result.hallucination.ungrounded_claims.length} unverified claim(s)</div>`;
        for (const claim of result.hallucination.ungrounded_claims.slice(0, 5)) {
            html += `<div class="halluc-claim">${escapeHtml(claim)}</div>`;
        }
        html += '</div>';
    }

    html += '</div>';
    div.innerHTML = html;
    container.appendChild(div);
    scrollToBottom();
}


// ─── Session Management ─────────────────────────────────────────────────────

function createNewSession(callback) {
    fetch("/api/session/new", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            currentSessionId = data.session_id;
            refreshSessionList();

            // Clear chat
            document.getElementById("messages").innerHTML = "";
            const welcome = document.getElementById("welcome-msg");
            if (welcome) welcome.style.display = "flex";

            if (callback) callback();
        })
        .catch(err => console.error("Failed to create session:", err));
}

function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;

    // Close active SSE
    if (activeSource) {
        activeSource.close();
        activeSource = null;
    }
    stopTrace();
    hideTracePanel();
    isProcessing = false;
    document.getElementById("btn-send").disabled = false;

    currentSessionId = sessionId;

    // Update sidebar active state
    document.querySelectorAll(".session-item").forEach(el => {
        el.classList.toggle("active", el.dataset.sessionId === sessionId);
    });

    // Load history
    loadSessionHistory(sessionId);
}

function loadSessionHistory(sessionId) {
    const container = document.getElementById("messages");
    container.innerHTML = "";

    fetch(`/api/session/${sessionId}/history`)
        .then(r => r.json())
        .then(data => {
            const welcome = document.getElementById("welcome-msg");
            if (data.messages && data.messages.length > 0) {
                if (welcome) welcome.style.display = "none";
                data.messages.forEach(msg => {
                    if (msg.role !== "system") {
                        appendMessage(msg.role, msg.content);
                    }
                });
            } else {
                if (welcome) welcome.style.display = "flex";
            }
        })
        .catch(err => console.error("Failed to load history:", err));
}

function refreshSessionList() {
    fetch("/api/sessions")
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("session-list");
            if (!data.sessions || data.sessions.length === 0) {
                list.innerHTML = '<div class="session-empty">No sessions yet</div>';
                return;
            }
            list.innerHTML = data.sessions.map(s =>
                `<div class="session-item ${s.id === currentSessionId ? "active" : ""}"
                     data-session-id="${s.id}"
                     onclick="switchSession('${s.id}')">
                    <div class="session-label">${escapeHtml(s.topic_label)}</div>
                    <div class="session-meta">${s.message_count} msg</div>
                </div>`
            ).join("");
        });
}

document.getElementById("btn-new-session").addEventListener("click", () => {
    createNewSession();
});


// ─── Utilities ──────────────────────────────────────────────────────────────

function scrollToBottom() {
    const container = document.getElementById("chat-container");
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

function formatMarkdown(text) {
    if (!text) return "";
    // Simple markdown → HTML conversion
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Inline code
    html = html.replace(/`(.+?)`/g, "<code>$1</code>");
    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    // Unordered lists
    html = html.replace(/^\* (.+)$/gm, "<li>$1</li>");
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);
    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, "</p><p>");
    html = `<p>${html}</p>`;
    // Clean empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, "");
    // Single newlines → <br>
    html = html.replace(/\n/g, "<br>");

    return html;
}
