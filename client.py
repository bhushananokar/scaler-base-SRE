"""
Incident Response Environment Client.

Example:
    >>> with IncidentResponseEnv(base_url="http://localhost:8000") as client:
    ...     # Start task 1 (Easy OOM)
    ...     result = client.reset(task_id=1)
    ...     print(result.observation.content)  # incident briefing
    ...
    ...     # Investigate
    ...     result = client.step(IncidentAction(tool="list_alerts"))
    ...     print(result.observation.content)
    ...
    ...     result = client.step(IncidentAction(
    ...         tool="query_logs", service="payment-service", severity="error"
    ...     ))
    ...     print(result.observation.content)
    ...
    ...     # Fix
    ...     result = client.step(IncidentAction(
    ...         tool="execute_remediation", action_type="restart", target="payment-service"
    ...     ))
    ...     print(result.observation.content)
    ...
    ...     # Score
    ...     result = client.step(IncidentAction(
    ...         tool="declare_resolved",
    ...         root_cause="OOM in payment-service due to unbounded cache growth; restarted service"
    ...     ))
    ...     print(result.observation.content)   # reward breakdown
    ...     print(result.reward)                # final score 0.0-1.0
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import IncidentAction, IncidentObservation


class IncidentResponseEnv(EnvClient[IncidentAction, IncidentObservation, State]):
    """
    Client for the AI-Powered Incident Response Triage Environment.

    Three tasks available via reset(task_id=N):
        task_id=1  -- Easy:   Single service OOM restart
        task_id=2  -- Medium: Bad deploy rollback with red herring
        task_id=3  -- Hard:   Cascading DB connection leak, multi-service failure

    SRE tools (pass to step via IncidentAction.tool):
        list_alerts             -- see what's currently firing
        query_logs              -- read service logs (service, severity, keyword)
        query_metrics           -- check current metrics (service, metric)
        read_runbook            -- consult ops runbook (topic)
        check_deploy_history    -- see recent deploys (service)
        execute_remediation     -- apply a fix (action_type, target)
        update_status           -- post to incident channel (message)
        declare_resolved        -- end episode and get scored (root_cause)
    """

    def _step_payload(self, action: IncidentAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[IncidentObservation]:
        obs_data = payload.get("observation", {})
        observation = IncidentObservation(
            content=obs_data.get("content", ""),
            task_id=obs_data.get("task_id", 0),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
