"""
Inference Script — Incident Response Triage Environment
========================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The OpenAI-compatible API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Defaults (active HF inference endpoint):
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"

Usage:
    python inference.py
    python inference.py --task_id 1       # single task
    python inference.py --task_id all     # all 3 tasks (default)

STDOUT FORMAT (strictly followed for automated evaluation):
    [START] task=<name> env=incident_response_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from server.my_env_environment import IncidentResponseEnvironment
from models import IncidentAction

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")

MAX_STEPS              = 15
SUCCESS_SCORE_THRESHOLD = 0.50   # score >= this => success

TASK_NAMES = {
    1: "oom-incident",
    2: "bad-deploy-incident",
    3: "cascade-incident",
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
                    "action_type": {"type": "string", "description": "'restart', 'rollback', 'scale', or 'flush_cache'"},
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
6. Call declare_resolved with a precise, specific root cause description.

Rules:
- Never restart a service before investigating its logs and metrics.
- In a cascading failure, fix the source service first, then the shared resource.
- Verify metrics after remediation.
- Be specific in declare_resolved: name the service, version, bug, and action taken.
"""

# ---------------------------------------------------------------------------
# Logging helpers (strict format required by evaluator)
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
    """Compact string representation for the [STEP] action field."""
    if not args:
        return f"{tool_name}()"
    parts = ",".join(f"{k}={v}" for k, v in args.items())
    return f"{tool_name}({parts})"


# ---------------------------------------------------------------------------
# LLM call (OpenAI client, tool-calling)
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, messages: list) -> Optional[dict]:
    """Call the LLM and return {'content': str, 'tool_calls': list}."""
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
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(client: OpenAI, task_id: int) -> dict:
    task_name = TASK_NAMES[task_id]
    log_start(task=task_name, model=MODEL_NAME)

    env = IncidentResponseEnvironment()
    obs = env.reset(task_id=task_id)

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

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    done = False
    last_error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            response = call_llm(client, messages)
            if response is None:
                last_error = "LLM returned no response"
                log_step(step, "none()", 0.0, False, last_error)
                rewards.append(0.0)
                steps_taken = step
                break

            tool_calls = response.get("tool_calls", [])
            content = response.get("content", "")

            if not tool_calls:
                # Model stopped calling tools — treat as stuck
                last_error = "no_tool_call"
                log_step(step, "none()", 0.0, False, last_error)
                rewards.append(0.0)
                steps_taken = step
                break

            # Build assistant message (with tool_calls for OpenAI history)
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

            # Execute each tool call against the environment
            tool_result_msgs = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}

                try:
                    action = IncidentAction(tool=tool_name, **tool_args)
                    result_obs = env.step(action)
                    result_str = result_obs.content
                    step_reward = result_obs.reward
                    step_done = result_obs.done
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
                    score = step_reward          # final reward at declare_resolved
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
    parser = argparse.ArgumentParser(description="Incident Response — Baseline Inference")
    parser.add_argument("--task_id", default="all", help="1, 2, 3, or 'all'")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = [1, 2, 3] if args.task_id == "all" else [int(args.task_id)]

    results = []
    for tid in task_ids:
        result = run_episode(client, tid)
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
