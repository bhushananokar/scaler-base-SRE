"""
Interactive demo UI for the Incident Response Triage environment.

Mounted at /ui on the FastAPI app.
Two modes:
  - Watch AI Agent  : an LLM solves the incident live, step by step
  - Try It Yourself : manual tool-by-tool exploration
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Generator

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.my_env_environment import IncidentResponseEnvironment  # noqa: E402
from models import IncidentAction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_INFO = {
    1: {
        "label": "Task 1 — Easy   | 🟢 OOM Heap Exhaustion",
        "brief": (
            "**Scenario:** `payment-service` is crashing with OutOfMemoryError. "
            "Transactions are failing. Customers cannot pay.\n\n"
            "**Your goal:** Find which service is the root cause and fix it."
        ),
        "hint": "💡 Start with **list_alerts**, then **query_logs** on payment-service.",
    },
    2: {
        "label": "Task 2 — Medium | 🟡 Bad Deployment Rollback",
        "brief": (
            "**Scenario:** A recent deployment broke `api-gateway`. "
            "All routes are returning 500 errors.\n\n"
            "**Your goal:** Identify the bad deployment and roll it back."
        ),
        "hint": "💡 Use **check_deploy_history** after checking logs.",
    },
    3: {
        "label": "Task 3 — Hard   | 🔴 Cascading DB Connection Leak",
        "brief": (
            "**Scenario:** Multiple services are down simultaneously. "
            "The shared database connection pool is exhausted.\n\n"
            "**Your goal:** Find the leaking service and fix the cascade."
        ),
        "hint": "💡 Check metrics for `db_connections_acquired` — the leaking service has an anomalous rate.",
    },
    4: {
        "label": "Task 4 — Medium | 🟡 TLS Certificate CN Mismatch",
        "brief": (
            "**Scenario:** `checkout-service` cannot talk to `payments-gateway`. "
            "SSL handshake failures everywhere.\n\n"
            "**Your goal:** Find the config mismatch and update it."
        ),
        "hint": "💡 Use **check_deploy_history** on payments-gateway, then **update_config**.",
    },
    5: {
        "label": "Task 5 — Hard   | 🔴 Alert Storm / Consumer Deadlock",
        "brief": (
            "**Scenario:** 11 alerts firing at once. `message-queue` is filling up. "
            "Producer services are backing up.\n\n"
            "**Your goal:** Find the deadlocked consumer and clear the queue."
        ),
        "hint": "💡 Check `notification-service` logs and deploy history first.",
    },
}

TOOL_CONFIGS = {
    "list_alerts":          {"params": [], "desc": "List all currently firing alerts — always start here"},
    "query_logs":           {"params": ["service", "severity", "keyword"], "desc": "Read recent logs for a service"},
    "query_metrics":        {"params": ["service", "metric"], "desc": "Check live metrics for a service"},
    "read_runbook":         {"params": ["topic"], "desc": "Consult the operations runbook"},
    "check_deploy_history": {"params": ["service"], "desc": "See recent deployments for a service"},
    "execute_remediation":  {"params": ["action_type", "target"], "desc": "Apply a fix (restart / rollback / scale / update_config / flush_cache)"},
    "update_status":        {"params": ["message"], "desc": "Post a status update to the incident channel"},
    "declare_resolved":     {"params": ["root_cause"], "desc": "Close the incident and get your final score"},
}

SEVERITY_CHOICES = ["all", "error", "warn"]
ACTION_TYPE_CHOICES = ["restart", "rollback", "scale", "update_config", "flush_cache"]

# LLM defaults (HF inference router — same as inference.py)
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
DEFAULT_MODEL    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer responding to a production incident.

Workflow:
1. Call list_alerts first.
2. Investigate: query_logs, query_metrics, read_runbook, check_deploy_history.
3. Find the ROOT CAUSE — not just symptoms.
4. Execute the correct remediation.
5. Post a status update.
6. Verify metrics, then call declare_resolved with a precise root cause.

Rules:
- Never remediate before investigating.
- In cascading failures, fix the SOURCE service, not downstream victims.
- For config/TLS issues use update_config, not restart.
- Be specific in declare_resolved: service, version, bug, action taken.
"""

AI_TOOLS = [
    {"type": "function", "function": {"name": "list_alerts", "description": "List all firing alerts.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "query_logs", "description": "Query logs for a service.", "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "severity": {"type": "string"}, "keyword": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "query_metrics", "description": "Check metrics for a service.", "parameters": {"type": "object", "properties": {"service": {"type": "string"}, "metric": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "read_runbook", "description": "Consult operations runbook.", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {"name": "check_deploy_history", "description": "Check deployment history.", "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {"name": "execute_remediation", "description": "Execute a remediation.", "parameters": {"type": "object", "properties": {"action_type": {"type": "string"}, "target": {"type": "string"}}, "required": ["action_type", "target"]}}},
    {"type": "function", "function": {"name": "update_status", "description": "Post a status update.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    {"type": "function", "function": {"name": "declare_resolved", "description": "Declare the incident resolved. Triggers scoring.", "parameters": {"type": "object", "properties": {"root_cause": {"type": "string"}}, "required": ["root_cause"]}}},
]

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _reward_badge(reward: float | None, done: bool) -> str:
    if reward is None:
        return "<span style='color:#888'>—</span>"
    if done:
        pct = int(reward * 100)
        color = "#22c55e" if reward >= 0.6 else "#f59e0b" if reward >= 0.4 else "#ef4444"
        return f"<strong style='color:{color};font-size:1.1em'>🏁 {reward:.3f} ({pct}%)</strong>"
    color = "#22c55e" if reward > 0 else "#ef4444" if reward < 0 else "#888"
    sign  = "+" if reward >= 0 else ""
    return f"<span style='color:{color}'>{sign}{reward:.4f}</span>"


def _tool_badge(tool: str) -> str:
    colors = {
        "list_alerts":          "#6366f1",
        "query_logs":           "#0ea5e9",
        "query_metrics":        "#0ea5e9",
        "read_runbook":         "#8b5cf6",
        "check_deploy_history": "#8b5cf6",
        "execute_remediation":  "#f59e0b",
        "update_status":        "#10b981",
        "declare_resolved":     "#22c55e",
    }
    color = colors.get(tool, "#64748b")
    return f"<code style='background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em'>{tool}</code>"


def _args_str(tool: str, kwargs: dict) -> str:
    filtered = {k: v for k, v in kwargs.items() if k != "tool" and v}
    if not filtered:
        return "()"
    parts = ", ".join(f"{k}=<em>{v}</em>" for k, v in filtered.items())
    return f"({parts})"


def build_steps_html(steps: list[dict]) -> str:
    if not steps:
        return "<p style='color:#888;text-align:center;padding:20px'>No steps yet — run the agent or execute a tool.</p>"

    rows = ""
    cumulative = 0.0
    for s in steps:
        r = s.get("reward")
        done = s.get("done", False)
        if isinstance(r, (int, float)):
            cumulative += r
        n     = s["step"]
        tool  = s["tool"]
        args  = s.get("args", {})
        badge = _reward_badge(r, done)
        cum_str = f"<span style='color:#94a3b8'>{cumulative:.4f}</span>"
        row_bg = "#1a2a1a" if done else ""
        rows += (
            f"<tr style='background:{row_bg}'>"
            f"<td style='padding:6px 10px;text-align:center;color:#94a3b8'>{n}</td>"
            f"<td style='padding:6px 10px'>{_tool_badge(tool)}{_args_str(tool, args)}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{badge}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{cum_str}</td>"
            f"</tr>"
        )

    return f"""
<div style='overflow-x:auto'>
<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:0.9em'>
  <thead>
    <tr style='border-bottom:1px solid #334155;color:#64748b;text-transform:uppercase;font-size:0.75em'>
      <th style='padding:6px 10px'>Step</th>
      <th style='padding:6px 10px;text-align:left'>Tool &amp; Args</th>
      <th style='padding:6px 10px;text-align:right'>Reward</th>
      <th style='padding:6px 10px;text-align:right'>Cumulative</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>
"""


def build_score_html(steps: list[dict]) -> str:
    terminal = [s for s in steps if s.get("done")]
    if not terminal:
        return ""
    score = terminal[-1].get("reward", 0.0)
    pct   = int(score * 100)
    if score >= 0.7:
        label, color, emoji = "Excellent", "#22c55e", "✅"
    elif score >= 0.5:
        label, color, emoji = "Good", "#f59e0b", "✅"
    elif score >= 0.3:
        label, color, emoji = "Partial", "#f97316", "⚠️"
    else:
        label, color, emoji = "Needs work", "#ef4444", "❌"

    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return f"""
<div style='border:2px solid {color};border-radius:8px;padding:16px;margin-top:8px;text-align:center'>
  <div style='font-size:2em;margin-bottom:4px'>{emoji} {label}</div>
  <div style='font-family:monospace;font-size:1.1em;color:{color}'>{bar}</div>
  <div style='font-size:1.4em;margin-top:6px;color:{color}'>
    <strong>Final Score: {score:.3f} / 1.0 &nbsp;({pct}%)</strong>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# AI agent runner
# ---------------------------------------------------------------------------

def _friendly_llm_error(exc: Exception) -> str:
    msg = str(exc)
    if "401" in msg or "Invalid username or password" in msg or "Unauthorized" in msg:
        return (
            "**Invalid or missing API token (401).**\n\n"
            "Your HuggingFace token was rejected.  \n"
            "• Make sure you copied the full token (starts with `hf_`)  \n"
            "• Get a new one at https://huggingface.co/settings/tokens"
        )
    if "402" in msg or "depleted" in msg or "credits" in msg:
        return (
            "**HuggingFace inference credits exhausted (402).**\n\n"
            "Your free monthly quota is used up.  \n"
            "• Upgrade to HF Pro, or  \n"
            "• Use an OpenAI key instead: set `API_BASE_URL=https://api.openai.com/v1` and paste an `sk-...` key above."
        )
    if "403" in msg or "Forbidden" in msg:
        return "**Access forbidden (403).** Your token may not have inference permissions. Use a token with at least *read* scope."
    return f"**LLM error:** {msg}"


def _call_llm(client, messages: list) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=AI_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.1,
        )
    except Exception as exc:
        return {"error": _friendly_llm_error(exc)}
    msg = resp.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    return {"content": msg.content or "", "tool_calls": tool_calls}


def run_ai_agent(task_id: int) -> Generator:
    """Generator — yields (steps_html, obs_text, score_html, status) after each step."""
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
    if not key:
        msg = "**Server not configured.** Set OPENAI_API_KEY (or HF_TOKEN) as a Space secret."
        yield "<p style='color:#f87171;padding:20px'>Server configuration error — API key not set.</p>", msg, "", "❌ Server config error"
        return
    client = OpenAI(base_url=DEFAULT_API_BASE, api_key=key)

    env   = IncidentResponseEnvironment()
    obs   = env.reset(task_id=task_id)
    steps: list[dict] = []
    max_steps = 20

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"INCIDENT BRIEFING:\n{obs.content}\n\nBegin. Call list_alerts first."},
    ]

    status = "🤖 Agent is thinking..."
    yield build_steps_html(steps), obs.content, "", status

    for step_n in range(1, max_steps + 1):
        response = _call_llm(client, messages)
        if response is None or "error" in response:
            err = response.get("error", "Unknown error") if response else "No response"
            status = f"❌ LLM error: {err}"
            yield build_steps_html(steps), f"Error: {err}", build_score_html(steps), status
            return

        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            status = "⚠️ Agent stopped calling tools."
            yield build_steps_html(steps), response.get("content", ""), build_score_html(steps), status
            return

        # Add assistant message to history
        messages.append({
            "role": "assistant",
            "content": response["content"],
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in tool_calls
            ],
        })

        tool_results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
            known = set(IncidentAction.model_fields.keys())
            safe_args = {k: v for k, v in tool_args.items() if k in known}

            try:
                action   = IncidentAction(tool=tool_name, **safe_args)
                result   = env.step(action)
                obs_text = result.observation.content if hasattr(result, "observation") else result.content
                reward   = result.observation.reward  if hasattr(result, "observation") else result.reward
                done     = result.observation.done    if hasattr(result, "observation") else result.done
            except Exception as exc:
                obs_text = f"Error: {exc}"
                reward   = None
                done     = False

            steps.append({
                "step":   step_n,
                "tool":   tool_name,
                "args":   safe_args,
                "reward": reward,
                "done":   done,
                "obs":    obs_text,
            })

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": obs_text,
            })

            status = f"🔄 Step {step_n} — {tool_name}()"

            yield build_steps_html(steps), obs_text, build_score_html(steps), status

            if done:
                status = "✅ Incident resolved!"
                yield build_steps_html(steps), obs_text, build_score_html(steps), status
                return

        messages.extend(tool_results)

    status = f"⏱️ Reached step limit ({max_steps})."
    yield build_steps_html(steps), steps[-1]["obs"] if steps else "", build_score_html(steps), status


# ---------------------------------------------------------------------------
# Manual step executor
# ---------------------------------------------------------------------------

def manual_step(
    env_holder: dict | None,
    steps: list,
    tool: str,
    service: str,
    severity: str,
    keyword: str,
    metric: str,
    topic: str,
    action_type: str,
    target: str,
    message: str,
    root_cause: str,
) -> tuple:
    # Always returns exactly 6 values:
    # (env_holder, steps, steps_html, obs_text, score_html, status)
    if env_holder is None:
        msg = "⚠️ Start an episode first — click **Start Episode** above."
        return None, steps, build_steps_html(steps), msg, build_score_html(steps), msg

    env: IncidentResponseEnvironment = env_holder["env"]
    step_n = len(steps) + 1

    kwargs: dict = {"tool": tool}
    if service:     kwargs["service"]     = service
    if severity and severity != "all": kwargs["severity"] = severity
    if keyword:     kwargs["keyword"]     = keyword
    if metric and metric != "all":    kwargs["metric"]   = metric
    if topic:       kwargs["topic"]       = topic
    if action_type: kwargs["action_type"] = action_type
    if target:      kwargs["target"]      = target
    if message:     kwargs["message"]     = message
    if root_cause:  kwargs["root_cause"]  = root_cause

    try:
        action   = IncidentAction(**kwargs)
        obs      = env.step(action)
        obs_text = obs.content
        reward   = obs.reward
        done     = obs.done
    except Exception as exc:
        err = f"Error: {exc}"
        return env_holder, steps, build_steps_html(steps), err, build_score_html(steps), f"❌ {exc}"

    steps = steps + [{
        "step":   step_n,
        "tool":   tool,
        "args":   {k: v for k, v in kwargs.items() if k != "tool"},
        "reward": reward,
        "done":   done,
        "obs":    obs_text,
    }]

    status = "✅ Incident resolved! Check your final score below." if done else f"Step {step_n} done — pick next tool."
    return env_holder, steps, build_steps_html(steps), obs_text, build_score_html(steps), status


def start_manual_episode(task_id: int) -> tuple:
    env = IncidentResponseEnvironment()
    obs = env.reset(task_id=task_id)
    return {"env": env}, [], build_steps_html([]), obs.content, "", "✅ Episode started — select a tool and execute."


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container { background: #0f172a !important; color: #e2e8f0 !important; }
.gr-button-primary { background: #6366f1 !important; border: none !important; }
.gr-button { border-radius: 6px !important; }
h1, h2, h3 { color: #f1f5f9 !important; }
.gr-textbox textarea, .gr-dropdown select { background: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important; }
.gr-markdown { color: #cbd5e1 !important; }
footer { display: none !important; }
"""


def build_demo() -> gr.Blocks:
    task_labels = [info["label"] for info in TASK_INFO.values()]
    task_map    = {info["label"]: tid for tid, info in TASK_INFO.items()}

    with gr.Blocks(title="Incident Response AI Demo") as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            """
# 🚨 AI-Powered Incident Response — Live Demo
**Watch an LLM agent triage real production incidents, step by step.**
Each tool call earns (or costs) reward. The agent scores 0–1 at resolution.
"""
        )

        with gr.Tabs():

            # ── Tab 1: Watch AI ──────────────────────────────────────────
            with gr.Tab("🤖 Watch AI Solve It"):
                gr.Markdown(
                    "Select a task and hit **▶ Run AI Agent**. "
                    "The agent will investigate and resolve the incident autonomously — no setup needed."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        ai_task_dd = gr.Dropdown(
                            choices=task_labels,
                            value=task_labels[0],
                            label="Select Incident Task",
                            interactive=True,
                        )
                        ai_task_desc = gr.Markdown(TASK_INFO[1]["brief"])
                        ai_hint      = gr.Markdown(TASK_INFO[1]["hint"])

                        ai_run_btn = gr.Button("▶  Run AI Agent", variant="primary", size="lg")
                        ai_status  = gr.Markdown("_Ready — pick a task and click Run._")

                    with gr.Column(scale=2):
                        ai_steps_html = gr.HTML(
                            label="Live Step Log",
                            value="<p style='color:#64748b;text-align:center;padding:40px'>Steps will appear here as the agent works.</p>",
                        )

                ai_obs_box    = gr.Textbox(label="Latest Tool Output", lines=10, interactive=False, buttons=["copy"])
                ai_score_html = gr.HTML()

                # Update task description when dropdown changes
                def _update_ai_desc(label):
                    tid = task_map[label]
                    return TASK_INFO[tid]["brief"], TASK_INFO[tid]["hint"]

                ai_task_dd.change(_update_ai_desc, inputs=[ai_task_dd], outputs=[ai_task_desc, ai_hint])

                # Run agent
                def _run_agent(label):
                    tid = task_map[label]
                    for steps_html, obs_text, score_html, status in run_ai_agent(tid):
                        yield steps_html, obs_text, score_html, status

                ai_run_btn.click(
                    _run_agent,
                    inputs=[ai_task_dd],
                    outputs=[ai_steps_html, ai_obs_box, ai_score_html, ai_status],
                )

            # ── Tab 2: Manual ────────────────────────────────────────────
            with gr.Tab("🧑‍💻 Try It Yourself"):
                gr.Markdown(
                    "Pick a task, start an episode, then call tools one by one. "
                    "Each call shows the result and your running reward."
                )

                m_env_state   = gr.State(None)
                m_steps_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1):
                        m_task_dd  = gr.Dropdown(choices=task_labels, value=task_labels[0], label="Select Task", interactive=True)
                        m_task_desc = gr.Markdown(TASK_INFO[1]["brief"])
                        m_hint      = gr.Markdown(TASK_INFO[1]["hint"])
                        m_start_btn = gr.Button("🔄  Start Episode", variant="primary")
                        m_status    = gr.Markdown("_Select a task and click Start Episode._")

                    with gr.Column(scale=2):
                        m_steps_html = gr.HTML(
                            value="<p style='color:#64748b;text-align:center;padding:40px'>Start an episode to begin.</p>"
                        )

                gr.Markdown("### Execute a Tool")
                gr.Markdown(
                    "Choose a tool from the dropdown. Fill in only the fields that tool needs "
                    "(see the description for guidance), then click **Execute**."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        m_tool_dd = gr.Dropdown(
                            choices=list(TOOL_CONFIGS.keys()),
                            value="list_alerts",
                            label="Tool",
                            interactive=True,
                        )
                        m_tool_desc = gr.Markdown(f"_{TOOL_CONFIGS['list_alerts']['desc']}_")

                    with gr.Column(scale=2):
                        with gr.Row():
                            m_service = gr.Textbox(label="service", placeholder="e.g. payment-service", lines=1)
                            m_severity = gr.Dropdown(choices=SEVERITY_CHOICES, value="all", label="severity")
                        with gr.Row():
                            m_keyword = gr.Textbox(label="keyword", placeholder="optional log keyword", lines=1)
                            m_metric  = gr.Textbox(label="metric",  placeholder="e.g. error_rate or all", lines=1)
                        with gr.Row():
                            m_topic   = gr.Textbox(label="topic",       placeholder="e.g. payment-service", lines=1)
                            m_action_type = gr.Dropdown(choices=ACTION_TYPE_CHOICES, label="action_type", value="restart")
                        with gr.Row():
                            m_target  = gr.Textbox(label="target",      placeholder="service to remediate", lines=1)
                            m_message = gr.Textbox(label="message",     placeholder="status update text", lines=1)
                        m_root_cause = gr.Textbox(
                            label="root_cause (for declare_resolved)",
                            placeholder="Describe root cause + action taken. Be specific: service, version, fix.",
                            lines=2,
                        )

                m_exec_btn     = gr.Button("⚡  Execute Tool", variant="primary")
                m_obs_box      = gr.Textbox(label="Tool Output", lines=10, interactive=False, buttons=["copy"])
                m_score_html   = gr.HTML()

                # Update task description
                def _update_m_desc(label):
                    tid = task_map[label]
                    return TASK_INFO[tid]["brief"], TASK_INFO[tid]["hint"]

                m_task_dd.change(_update_m_desc, inputs=[m_task_dd], outputs=[m_task_desc, m_hint])

                # Update tool description
                def _update_tool_desc(tool):
                    return f"_{TOOL_CONFIGS[tool]['desc']}_"

                m_tool_dd.change(_update_tool_desc, inputs=[m_tool_dd], outputs=[m_tool_desc])

                # Start episode
                def _start(label):
                    tid = task_map[label]
                    env_h, steps, steps_html, obs_text, score_html, status = start_manual_episode(tid)
                    return env_h, steps, steps_html, obs_text, score_html, status

                m_start_btn.click(
                    _start,
                    inputs=[m_task_dd],
                    outputs=[m_env_state, m_steps_state, m_steps_html, m_obs_box, m_score_html, m_status],
                )

                # Execute tool
                def _exec(env_h, steps, tool, service, severity, keyword, metric, topic, action_type, target, message, root_cause):
                    env_h2, steps2, steps_html, obs_text, score_html, status = manual_step(
                        env_h, steps, tool, service, severity, keyword, metric,
                        topic, action_type, target, message, root_cause,
                    )
                    return env_h2, steps2, steps_html, obs_text, score_html, status

                m_exec_btn.click(
                    _exec,
                    inputs=[m_env_state, m_steps_state, m_tool_dd, m_service, m_severity, m_keyword,
                            m_metric, m_topic, m_action_type, m_target, m_message, m_root_cause],
                    outputs=[m_env_state, m_steps_state, m_steps_html, m_obs_box, m_score_html, m_status],
                )

        # ── Footer ──────────────────────────────────────────────────────
        gr.Markdown(
            """
---
**Tools:** `list_alerts` → `query_logs` / `query_metrics` → `read_runbook` / `check_deploy_history`
→ `execute_remediation` → `update_status` → `declare_resolved`

**Scoring:** 8-dimensional reward (situational awareness · diagnostic quality · remediation · time efficiency · communication · anti-patterns · epistemic quality · workflow coherence)
"""
        )

    return demo
