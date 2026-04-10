"""
Incident Response Environment — OpenEnv server implementation.

Each tool call returns both a result string AND a per-step reward delta.
The observation's `reward` field carries this per-step signal so RL agents
get immediate feedback rather than waiting for the terminal step.

At declare_resolved, `done=True` and `reward` = final normalized score (0–1).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IncidentAction, IncidentObservation
except ImportError:
    from models import IncidentAction, IncidentObservation

from src.oom_incident import OOMIncident
from src.deploy_incident import DeployIncident
from src.cascade_incident import CascadeIncident
from src.config_drift_incident import ConfigDriftIncident
from src.alert_storm_incident import AlertStormIncident


def _make_incident(task_id: int, seed: int):
    if task_id == 1:
        return OOMIncident(seed=seed)
    elif task_id == 2:
        return DeployIncident(seed=seed)
    elif task_id == 3:
        return CascadeIncident(seed=seed)
    elif task_id == 4:
        return ConfigDriftIncident(seed=seed)
    elif task_id == 5:
        return AlertStormIncident(seed=seed)
    raise ValueError(f"Unknown task_id {task_id}. Valid: 1–5")


class IncidentResponseEnvironment(Environment):
    """
    AI-Powered Incident Response Triage Environment.

    8-dimensional reward system with 29 signal components:
      D1 Situational Awareness  — alert ack, blast radius, investigation depth
      D2 Diagnostic Quality     — root cause ID, efficiency, coherence, red-herring resistance
      D3 Remediation Quality    — action correctness, ordering, collateral damage, fix verification
      D4 Time Efficiency        — MTTD, MTTR, SLA compliance
      D5 Communication          — update cadence, update quality, resolution accuracy
      D6 Anti-pattern Penalties — blind remediations, circular queries, red herring actions
      D7 Epistemic Quality      — Bayesian belief entropy reduction, final confidence, redundancy
      D8 Workflow Coherence     — SRE phase progression machine (OBSERVE→HYPOTHESIZE→DIAGNOSE→REMEDIATE→VERIFY)

    Per-step rewards: ~30% of total (immediate signal via observation.reward)
    Episode-end reward: ~70% of total (at done=True)

    Observation fields include workflow_phase and epistemic_confidence so agents
    can explicitly reason about their investigation state.

    Tasks:
        task_id=1  Easy        — OOM in payment-service
        task_id=2  Medium      — Bad deploy (api-gateway) with two red herrings
        task_id=3  Hard        — Cascading DB connection leak across 5 services
        task_id=4  Medium-Hard — Config drift: TLS cert CN mismatch in checkout-service
        task_id=5  Hard        — Alert storm: notification-service consumer deadlock (max 20 steps)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._incident = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: int = 1, seed: int = 42, **kwargs: Any) -> IncidentObservation:
        """
        Start a new incident episode.

        Args:
            task_id: 1=Easy OOM, 2=Medium Deploy, 3=Hard Cascade (default: 1)
            seed: Reproducibility seed (default: 42)
        """
        self._incident = _make_incident(task_id, seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        context = self._incident.get_initial_context()
        return IncidentObservation(
            content=context,
            task_id=task_id,
            step_count=0,
            done=False,
            reward=None,
            metadata={"status": "incident_opened", "task_id": task_id},
        )

    def step(self, action: IncidentAction) -> IncidentObservation:  # type: ignore[override]
        """
        Execute one SRE action. Returns per-step reward signal in observation.reward.
        At declare_resolved, done=True and reward=final_score.
        """
        if self._incident is None:
            return IncidentObservation(
                content="No active incident. Call reset() with task_id first.",
                done=False, reward=None, task_id=0, step_count=0,
            )

        self._state.step_count += 1
        tool = (action.tool or "").strip().lower()
        result, step_reward = self._dispatch(tool, action)

        # Collect epistemic + workflow state for observation
        wf_phase = self._incident.workflow_machine.phase
        ep_confidence = round(self._incident.belief_engine.confidence(), 4)

        done = self._incident.done
        if done:
            # step_reward is already the clamped final score from compute_final_reward()
            final_reward = max(0.001, min(0.999, float(step_reward)))
            return IncidentObservation(
                content=result,
                task_id=self._incident.get_task_id(),
                step_count=self._state.step_count,
                done=True,
                reward=final_reward,
                workflow_phase=wf_phase,
                epistemic_confidence=ep_confidence,
                metadata={"tool": tool, "final": True},
            )

        clamped_reward = max(0.001, min(0.999, round(step_reward, 4)))
        return IncidentObservation(
            content=result,
            task_id=self._incident.get_task_id(),
            step_count=self._state.step_count,
            done=False,
            reward=clamped_reward,
            workflow_phase=wf_phase,
            epistemic_confidence=ep_confidence,
            metadata={"tool": tool, "step_reward": clamped_reward},
        )

    def _dispatch(self, tool: str, action: IncidentAction) -> tuple[str, float]:
        inc = self._incident

        if tool == "list_alerts":
            return inc.list_alerts()

        elif tool == "query_logs":
            return inc.query_logs(
                service=action.service or "",
                severity=action.severity or "all",
                keyword=action.keyword or "",
            )

        elif tool == "query_metrics":
            return inc.query_metrics(
                service=action.service or "",
                metric=action.metric or "all",
            )

        elif tool == "read_runbook":
            return inc.read_runbook(topic=action.topic or "")

        elif tool == "check_deploy_history":
            return inc.check_deploy_history(service=action.service or "")

        elif tool == "execute_remediation":
            return inc.execute_remediation(
                action=action.action_type or "",
                target=action.target or "",
            )

        elif tool == "update_status":
            return inc.update_status(message=action.message or "")

        elif tool == "declare_resolved":
            reward, feedback = inc.declare_resolved(root_cause=action.root_cause or "")
            return feedback, reward

        else:
            valid = [
                "list_alerts", "query_logs", "query_metrics", "read_runbook",
                "check_deploy_history", "execute_remediation", "update_status", "declare_resolved",
            ]
            return f"Unknown tool '{tool}'. Valid: {valid}", 0.0

    @property
    def state(self) -> State:
        return self._state
