"""
Remote Inference Script — Incident Response Triage Environment
==============================================================
Identical agent logic to inference.py but uses the deployed HF Space
via IncidentResponseEnv (WebSocket client) instead of the local env.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The OpenAI-compatible API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        The deployed environment URL (default: HF Space URL).

Usage:
    python inference_remote.py
    python inference_remote.py --task_id 1       # single task
    python inference_remote.py --task_id all     # all 5 tasks (default)
    python inference_remote.py --env_url https://your-space.hf.space

STDOUT FORMAT:
    [START] task=<name> env=incident_response_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
import importlib, types

# client.py uses relative imports; load it with a package context
_pkg = types.ModuleType("_incident_pkg")
_pkg.__path__ = [os.path.dirname(__file__)]
_pkg.__package__ = "_incident_pkg"
import sys as _sys
_sys.modules["_incident_pkg"] = _pkg

import importlib.util as _ilu

def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path, submodule_search_locations=[])
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.dirname(__file__)
_models_mod = _load("_incident_pkg.models", os.path.join(_base, "models.py"))
_client_mod  = _load("_incident_pkg.client",  os.path.join(_base, "client.py"))

IncidentResponseEnv = _client_mod.IncidentResponseEnv
IncidentAction      = _models_mod.IncidentAction

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "placeholder")
DEFAULT_ENV_URL = "https://ghost14504-nrg-meta-submission.hf.space"

SUCCESS_SCORE_THRESHOLD = 0.50

TASK_NAMES = {
    1: "oom-incident",
    2: "bad-deploy-incident",
    3: "cascade-incident",
    4: "config-drift-incident",
    5: "alert-storm-incident",
}

TASK_MAX_STEPS = {
    1: 15, 2: 15, 3: 15, 4: 15, 5: 20,
}

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_alerts",
            "description": "List all currently firing alerts. Always start here.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": "Query recent logs for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service":  {"type": "string", "description": "Service name, e.g. 'payment-service'"},
                    "severity": {"type": "string", "description": "'error', 'warn', or 'all'"},
                    "keyword":  {"type": "string", "description": "Optional keyword filter"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": "Check current live metrics for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "metric":  {"type": "string", "description": "Metric name or 'all'"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_runbook",
            "description": "Consult the operations runbook for a service or issue type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "e.g. 'payment-service', 'db-pool', 'rollback', 'oom'"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_deploy_history",
            "description": "Check recent deployment history for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_remediation",
            "description": "Execute a remediation. Always investigate before remediating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "description": "'restart', 'rollback', 'scale', 'update_config', or 'flush_cache'"},
                    "target":      {"type": "string", "description": "Target service name"},
                },
                "required": ["action_type", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_status",
            "description": "Post a status update to the incident channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_resolved",
            "description": (
                "Declare the incident resolved. Triggers final scoring. "
                "Provide a precise root cause and fix description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root_cause": {
                        "type": "string",
                        "description": "Root cause + remediation taken. Be specific about service, version, and action.",
                    },
                },
                "required": ["root_cause"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer responding to a production incident.

Your workflow:
1. Call list_alerts first to see what is firing.
2. Investigate with query_logs, query_metrics, read_runbook, check_deploy_history.
3. Identify the ROOT CAUSE — not just symptoms. Look for the service or deploy causing the issue.
4. Execute the correct remediation(s) in the right order.
5. Post a status update describing what you found and fixed.
6. Verify metrics after remediation, then call declare_resolved with a precise root cause description.

Rules:
- Never remediate before investigating logs and metrics.
- In a cascading failure (multiple services failing with the same error), the root cause is
  the SHARED DEPENDENCY or the SERVICE OVERLOADING it — NOT the failing downstream services.
  Check which service has anomalous rates (e.g. DBConnectionRate >> baseline) in the alerts.
  Fix the source first, then the shared resource. Restarting symptoms gives only temporary relief.
- When alerts show a service with an anomalous RATE alert (e.g. DBConnectionRate, MessageRate),
  that service is likely the root cause — investigate it with query_logs and query_metrics.
- For config/TLS issues: use update_config as the remediation action, not restart or rollback.
- Be specific in declare_resolved: name the service, version, bug, and action taken.
"""

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=incident_response_env model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_str(tool_name: str, args: dict) -> str:
    if not args:
        return f"{tool_name}()"
    parts = ",".join(f"{k}={v}" for k, v in args.items())
    return f"{tool_name}({parts})"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, messages: list) -> Optional[dict]:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.2,
        )
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return None

    msg = response.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id":        tc.id,
                "name":      tc.function.name,
                "arguments": args,
            })
    return {"content": msg.content or "", "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Episode runner — uses remote EnvClient
# ---------------------------------------------------------------------------
def run_episode(llm_client: OpenAI, task_id: int, env_url: str) -> dict:
    task_name = TASK_NAMES[task_id]
    max_steps = TASK_MAX_STEPS.get(task_id, 15)
    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    done = False

    with IncidentResponseEnv(base_url=env_url).sync() as env:
        reset_result = env.reset(task_id=task_id)
        obs = reset_result.observation

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"INCIDENT BRIEFING:\n{obs.content}\n\n"
                    "Begin your investigation. Call list_alerts first."
                ),
            },
        ]

        try:
            for step in range(1, max_steps + 1):
                if done:
                    break

                response = call_llm(llm_client, messages)
                if response is None:
                    last_error = "LLM returned no response"
                    log_step(step, "none()", 0.0, False, last_error)
                    rewards.append(0.0)
                    steps_taken = step
                    break

                tool_calls = response.get("tool_calls", [])
                content = response.get("content", "")

                if not tool_calls:
                    last_error = "no_tool_call"
                    log_step(step, "none()", 0.0, False, last_error)
                    rewards.append(0.0)
                    steps_taken = step
                    break

                assistant_msg: dict = {"role": "assistant", "content": content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else tc["arguments"],
                        },
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)

                tool_result_msgs = []
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}

                    try:
                        known_fields = set(IncidentAction.model_fields.keys())
                        tool_args = {k: v for k, v in tool_args.items() if k in known_fields}
                        action = IncidentAction(tool=tool_name, **tool_args)
                        result = env.step(action)
                        result_str = result.observation.content
                        step_reward = result.reward or 0.0
                        step_done = result.done
                        err_str = None
                    except Exception as exc:
                        result_str = f"Error: {exc}"
                        step_reward = 0.0
                        step_done = False
                        err_str = str(exc)[:80]

                    action_repr = _action_str(tool_name, tool_args)
                    log_step(step, action_repr, step_reward, step_done, err_str)
                    rewards.append(step_reward)
                    steps_taken = step

                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

                    if step_done:
                        done = True
                        score = step_reward
                        success = score >= SUCCESS_SCORE_THRESHOLD
                        break

                messages.extend(tool_result_msgs)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "steps":     steps_taken,
        "score":     score,
        "success":   success,
        "rewards":   rewards,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Incident Response — Inference")
    parser.add_argument("--task_id",  default="all", help="1-5, or 'all'")
    parser.add_argument("--env_url",  default=os.getenv("ENV_URL", DEFAULT_ENV_URL),
                        help="Deployed environment base URL")
    args = parser.parse_args()

    print(f"[CONFIG] env_url={args.env_url} model={MODEL_NAME}", flush=True)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = [1, 2, 3, 4, 5] if args.task_id == "all" else [int(args.task_id)]

    results = []
    for tid in task_ids:
        result = run_episode(llm_client, tid, args.env_url)
        results.append(result)

    if len(results) > 1:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\n[SUMMARY] tasks={len(results)} avg_score={avg:.3f}", flush=True)
        for r in results:
            status = "success" if r["success"] else "failed"
            print(
                f"  task={r['task_name']} status={status} score={r['score']:.3f} steps={r['steps']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
