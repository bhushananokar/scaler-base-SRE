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
| 1 | `oom-incident` | Easy | **OOM Restart** — `payment-service` heap exhausted by unbounded `TransactionCache`. Red herring: `api-gateway` CPU+latency alert. |
| 2 | `bad-deploy-incident` | Medium | **Bad Deploy Rollback** — `api-gateway v2.3.1` removed lazy init, causing NPE on all routes. Two recent deploys — find the right one. |
| 3 | `cascade-incident` | Hard | **Cascading Failure** — `order-service v4.2.0` leaks DB connections, exhausting `db-pool`, taking down 3 downstream services. Ordering fix required. |
| 4 | `config-drift-incident` | Medium-Hard | **Config Drift** — `payments-gateway` TLS cert rotated (CN changed); `checkout-service` config stale. Fix is `update_config`, not restart. Red herrings: `redis-session` connection storm (downstream symptom) and `inventory-service` rollback noise. |
| 5 | `alert-storm-incident` | Hard | **Alert Storm** — 11 alerts; `notification-service v2.8.0` JPA deadlock stops all message consumption; `message-queue` fills (97%); 4 producers back up. Fix order: restart consumer THEN queue. 2 unrelated red herrings. |

## Reward System (8 Dimensions)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| **D1 Situational Awareness** | ~0.06–0.08 | Alert ack, blast radius mapping, investigation depth |
| **D2 Diagnostic Quality** | ~0.14–0.22 | Root cause ID, efficiency, red-herring resistance |
| **D3 Remediation Quality** | ~0.22–0.35 | Correct fix, right ordering, fix verification |
| **D4 Time Efficiency** | 0.10 | MTTD/MTTR — live metrics degrade every step |
| **D5 Communication** | 0.10 | Status update cadence and resolution accuracy |
| **D6 Anti-patterns** | 0.10–0.13 | Penalties: blind restart, circular queries, wrong service |
| **D7 Epistemic Quality** | 0.08 | Bayesian entropy reduction over root cause hypotheses |
| **D8 Workflow Coherence** | 0.07 | SRE phase-sequence adherence machine |

Per-step rewards (~30%) are returned on every `env.step()`. Episode-end reward (~70%) triggers at `declare_resolved`.

---

## Reward Design Innovation

### D7 — Epistemic Bayesian Reward Signal

The environment maintains a **hidden belief distribution** over a fixed set of competing root cause hypotheses per incident (e.g. `oom:payment-service`, `bad_deploy:payment-service`, `red_herring:api-gateway-cpu`). At episode start the prior is uniform — the agent knows nothing except the alert list.

Each investigation action (query_logs, query_metrics, read_runbook, check_deploy_history) triggers a **Bayesian update**: the engine multiplies the current distribution by hand-authored likelihood weights `P(observing this tool+result | hypothesis is true)`, then renormalizes. The **per-step reward is the Shannon entropy reduction** — how much the agent's uncertainty over root causes sharpened from this single action, normalized to [0, 1] and scaled to ~0.01–0.04.

Key properties:
- **Redundant queries earn near-zero reward**: if the same `tool:service` key was already called, likelihood weights are shrunk by 90%, so the posterior barely moves
- **Confidence gate**: `belief_engine.is_confident_enough(threshold=0.65)` is checked before remediation; acting before confidence is below this threshold counts as premature
- **Final confidence score** (episode-end): probability mass on the true hypothesis at `declare_resolved` time contributes to D7

### D8 — SRE Workflow Coherence Machine

A **finite state machine** tracks which phase of the SRE investigation workflow the agent is in:

```
OBSERVE → HYPOTHESIZE → DIAGNOSE → REMEDIATE → VERIFY
```

| Phase | Entry condition | Reward |
|-------|----------------|--------|
| HYPOTHESIZE | First query tool called | +0.02 |
| DIAGNOSE | Root cause service specifically investigated | +0.03 |
| REMEDIATE | Correct remediation applied after diagnosis | +0.03 |
| VERIFY | Metrics/logs queried after remediation | +0.02 |

Violations are penalized:
- Remediating from `OBSERVE` (no investigation): **−0.05**
- Declaring resolved before `VERIFY`: **−0.04** + `fix_verification` score zeroed in D3

The machine's current state is surfaced in every observation as `workflow_phase` (string) and `epistemic_confidence` (float), making the agent's own reasoning state explicit and learnable.

**Why this matters**: agents are rewarded not just for *what* they discover, but for *how methodically* they discover it. This prevents reward hacking where an agent guesses the correct fix immediately without evidence.

---

**Baseline scores (optimal path):**

| Task | Score | D7 Epistemic | D8 Workflow |
|------|-------|-------------|------------|
| T1 OOM | 0.784 | 0.051/0.080 | 0.063/0.070 |
| T2 Bad Deploy | 0.728 | 0.048/0.080 | 0.070/0.070 |
| T3 Cascade | 0.736 | 0.048/0.080 | 0.070/0.070 |
| T4 Config Drift | 0.777 | 0.053/0.080 | 0.070/0.070 |
| T5 Alert Storm | 0.683 | 0.053/0.100 | 0.070/0.070 |

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

# Run smoke tests (no API key needed) — all 5 tasks
python test_quick.py

# Run the baseline agent (single task or all)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --task_id all          # tasks 1–5
python inference.py --task_id 4            # T4 only
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
├── test_quick.py             # Smoke tests — optimal path for all 5 tasks
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
    ├── reward_engine.py      # 8D, 29-component reward system (D1–D8)
    ├── belief_engine.py      # D7: Epistemic Bayesian Reward Signal
    ├── workflow_machine.py   # D8: SRE Workflow Coherence Machine
    ├── metric_engine.py      # Live metric evolution per step
    ├── incident_base.py      # Base class wiring all engines
    ├── oom_incident.py          # Task 1 — OOM
    ├── deploy_incident.py       # Task 2 — Bad Deploy
    ├── cascade_incident.py      # Task 3 — Cascade
    ├── config_drift_incident.py # Task 4 — Config Drift (TLS cert CN mismatch)
    └── alert_storm_incident.py  # Task 5 — Alert Storm (consumer deadlock)
```
