# AI-Powered Incident Response Triage Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that trains AI agents to triage and resolve production incidents. The agent plays the role of an on-call SRE, using investigation tools to find the root cause and apply the correct remediation.

## Overview

Each episode is a production incident. The agent must:

1. **Observe** — read alerts, query logs, check metrics, consult runbooks
2. **Diagnose** — identify the root cause (not just symptoms)
3. **Act** — execute the correct remediation(s) in the right order
4. **Communicate** — post status updates and declare resolved with a precise root cause

## Tasks

| # | Name | Difficulty | Scenario |
|---|------|-----------|----------|
| 1 | `oom-incident` | Easy | **OOM Restart** — `payment-service` heap exhausted by unbounded `TransactionCache`. Red herring: `api-gateway` CPU alert. |
| 2 | `bad-deploy-incident` | Medium | **Bad Deploy Rollback** — `api-gateway v2.3.1` removed lazy init, causing NPE on all routes. Two recent deploys — find the right one. |
| 3 | `cascade-incident` | Hard | **Cascading Failure** — `order-service v4.2.0` leaks DB connections, exhausting `db-pool`, taking down 3 downstream services. Symptom-only restarts recur. |

## Reward System (6 Dimensions)

| Dimension | What it measures |
|-----------|-----------------|
| **D1 Situational Awareness** | Alert ack, blast radius mapping, investigation depth |
| **D2 Diagnostic Quality** | Root cause ID, efficiency, red-herring resistance |
| **D3 Remediation Quality** | Correct fix, right ordering, fix verification |
| **D4 Time Efficiency** | MTTD/MTTR — live metrics degrade every step |
| **D5 Communication** | Status update cadence and resolution accuracy |
| **D6 Anti-patterns** | Penalties: blind restart, circular queries, wrong service |

Per-step rewards (~30%) are returned on every `env.step()`. Episode-end reward (~70%) triggers at `declare_resolved`.

**Baseline scores (optimal path):**

| Task | Optimal | Blind Restart | Red Herring Trap |
|------|---------|--------------|-----------------|
| T1 OOM | 0.794 | 0.195 | — |
| T2 Bad Deploy | 0.732 | — | 0.283 |
| T3 Cascade | 0.734 | — | 0.000 |

## Action Space

All actions use `IncidentAction` with a `tool` field:

| Tool | Key Parameters | Purpose |
|------|---------------|---------|
| `list_alerts` | — | See all firing alerts |
| `query_logs` | `service`, `severity`, `keyword` | Read service logs |
| `query_metrics` | `service`, `metric` | Check current live metrics |
| `read_runbook` | `topic` | Consult the operations runbook |
| `check_deploy_history` | `service` | See recent deployments |
| `execute_remediation` | `action_type`, `target` | Apply fix: restart / rollback / scale / flush_cache |
| `update_status` | `message` | Post to incident channel |
| `declare_resolved` | `root_cause` | End episode and trigger scoring |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests (no API key needed)
python test_quick.py

# Run the baseline agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --task_id all
```

## Baseline Agent Output Format

```
[START] task=oom-incident env=incident_response_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=list_alerts() reward=0.03 done=false error=null
[STEP]  step=2 action=query_logs(service=payment-service,severity=error) reward=0.02 done=false error=null
[STEP]  step=3 action=execute_remediation(action_type=restart,target=payment-service) reward=0.02 done=false error=null
[STEP]  step=4 action=declare_resolved(root_cause=...) reward=0.77 done=true error=null
[END]   success=true steps=4 score=0.794 rewards=0.03,0.02,0.02,0.77
```

## Running the Server

```bash
# Development (from this directory)
uvicorn server.app:app --reload --port 8000

# API docs at http://localhost:8000/docs
```

## Docker

```bash
docker build -f server/Dockerfile -t incident-response-env .
docker run -p 8000:8000 incident-response-env
```

## Project Structure

```
my_env/
├── inference.py              # Baseline agent (OpenAI-compatible, strict log format)
├── test_quick.py             # Smoke tests — optimal path for all 3 tasks
├── models.py                 # IncidentAction + IncidentObservation Pydantic models
├── openenv.yaml              # OpenEnv spec
├── pyproject.toml
├── requirements.txt
├── server/
│   ├── app.py                # FastAPI server
│   ├── my_env_environment.py # IncidentResponseEnvironment
│   ├── Dockerfile
│   └── requirements.txt
└── src/
    ├── reward_engine.py      # 6D, 23-component reward system
    ├── metric_engine.py      # Live metric evolution per step
    ├── incident_base.py      # Base class wiring reward + metric engines
    ├── oom_incident.py       # Task 1 — OOM
    ├── deploy_incident.py    # Task 2 — Bad Deploy
    └── cascade_incident.py   # Task 3 — Cascade
```
