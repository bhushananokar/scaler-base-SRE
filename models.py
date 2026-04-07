"""
Data models for the Incident Response Environment.

IncidentAction covers all SRE tool calls via a single `tool` field selector.
IncidentObservation carries the tool result and episode state.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class IncidentAction(Action):
    """
    Action for the Incident Response Environment.

    Select which investigation or remediation tool to use via `tool`.
    Supply the relevant parameter fields for the chosen tool.

    Tools and their required fields:
        list_alerts             -- (no extra fields needed)
        query_logs              -- service, [severity], [keyword]
        query_metrics           -- service, [metric]
        read_runbook            -- topic
        check_deploy_history    -- service
        execute_remediation     -- action_type, target
        update_status           -- message
        declare_resolved        -- root_cause
    """

    tool: str = Field(
        ...,
        description=(
            "Which SRE tool to invoke. One of: list_alerts, query_logs, query_metrics, "
            "read_runbook, check_deploy_history, execute_remediation, update_status, declare_resolved"
        ),
    )
    service: Optional[str] = Field(
        default=None,
        description="Service name for query_logs, query_metrics, check_deploy_history "
                    "(e.g. 'payment-service', 'api-gateway', 'order-service', 'db-pool')",
    )
    severity: Optional[str] = Field(
        default="all",
        description="Log severity filter for query_logs: 'error', 'warn', or 'all'",
    )
    keyword: Optional[str] = Field(
        default="",
        description="Keyword to filter log lines in query_logs (e.g. 'timeout', 'OOM')",
    )
    metric: Optional[str] = Field(
        default="all",
        description="Metric name for query_metrics, or 'all' for all metrics. "
                    "Common: error_rate, latency_p99, memory_usage, connections_used, db_connections_acquired",
    )
    topic: Optional[str] = Field(
        default=None,
        description="Runbook topic for read_runbook "
                    "(e.g. 'payment-service', 'api-gateway', 'db-pool', 'rollback', 'cascade', 'oom')",
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Remediation action for execute_remediation: 'restart', 'rollback', 'scale', 'flush_cache'",
    )
    target: Optional[str] = Field(
        default=None,
        description="Target service or resource for execute_remediation "
                    "(e.g. 'payment-service', 'api-gateway', 'db-pool')",
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message text for update_status",
    )
    root_cause: Optional[str] = Field(
        default=None,
        description=(
            "Root cause description for declare_resolved. Be specific: name the service, "
            "cause type, and fix. "
            "Example: 'order-service v4.2.0 DB connection leak exhausted db-pool; "
            "restarted order-service and scaled db-pool'"
        ),
    )


class IncidentObservation(Observation):
    """
    Observation from the Incident Response Environment.

    Contains the text result of the last tool call and current episode state.
    When done=True, reward contains the final score (0.0 to 1.0).
    """

    content: str = Field(
        default="",
        description="Text result of the last tool call, or initial incident briefing on reset",
    )
    task_id: int = Field(
        default=0,
        description="Current task: 1=Easy OOM, 2=Medium Deploy, 3=Hard Cascade",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken in the current episode",
    )
