"""
Task 1: OOM Restart (Easy)

Scenario: payment-service has an unbounded TransactionCache that leaks memory.
Memory grows ~1.8%/step. At 100% the JVM crashes and downstream errors cascade.

Red herrings (strengthened):
  - api-gateway shows elevated CPU (68%) AND elevated latency (1850ms) in metrics —
    looks like a candidate for 1-2 investigation steps until logs show no errors.
  - payment-service v3.8.2 WARN log mentions "TransactionCache warm-start (eviction=disabled)"
    making the deploy look like the trigger — but the runbook reveals eviction was disabled
    in v3.7.0 (2 weeks earlier), not in this deploy.

Optimal path (5–7 steps):
  list_alerts -> query_logs(payment-service, error) -> read_runbook(payment-service)
  -> execute_remediation(restart, payment-service) -> query_metrics(payment-service) [verify]
  -> update_status(...) -> declare_resolved(...)

IncidentConfig golden actions: [("restart", "payment-service")]
"""

from .incident_base import BaseIncident
from .reward_engine import RewardEngine, IncidentConfig
from .metric_engine import make_oom_metric_engine
from .belief_engine import (
    BeliefEngine,
    OOM_HYPOTHESIS_SPACE, OOM_TRUE_HYPOTHESIS, OOM_LIKELIHOOD_TABLE,
)
from .workflow_machine import WorkflowMachine

# ---------------------------------------------------------------------------
# Static data: alerts, logs, runbooks, deploy history
# ---------------------------------------------------------------------------

_INITIAL_ALERTS = """
=== FIRING ALERTS ===
[CRITICAL] payment-service | MemoryUsage    | 97%      | threshold: 90%    | duration: 8m  | trending UP
[HIGH]     payment-service | RequestLatency | p99=4200ms | threshold: 1000ms | duration: 6m
[MEDIUM]   payment-service | ErrorRate      | 3.2%     | threshold: 5%     | duration: 3m  | trending UP
[LOW]      api-gateway     | CPUUsage       | 68%      | threshold: 70%    | duration: 2m  | flapping
"""

_LOGS = {
    "payment-service": {
        "error": (
            "[ERROR] 2024-01-15 14:32:01 payment-service java.lang.OutOfMemoryError: Java heap space\n"
            "    at com.payments.cache.TransactionCache.store(TransactionCache.java:142)\n"
            "    at com.payments.service.PaymentProcessor.process(PaymentProcessor.java:87)\n"
            "    TransactionCache.size=937MB heap.max=1024MB\n"
            "[ERROR] 2024-01-15 14:31:58 payment-service java.lang.OutOfMemoryError: Java heap space (x12 last 60s)\n"
            "[ERROR] 2024-01-15 14:31:45 payment-service GC overhead limit exceeded - heap exhausted\n"
            "[WARN]  2024-01-15 14:31:30 payment-service Memory 94% - cache growing unexpectedly\n"
            "[WARN]  2024-01-15 14:30:10 payment-service TransactionCache.size growing: 760MB -> 892MB in 5min\n"
        ),
        "warn": (
            "[WARN]  2024-01-15 14:31:30 payment-service Memory 94% - approaching limit\n"
            "[WARN]  2024-01-15 14:30:10 payment-service TransactionCache growing unexpectedly: 892MB (limit 1024MB)\n"
            "[WARN]  2024-01-15 14:28:55 payment-service Cache eviction disabled since v3.7.0 (config change)\n"
            "[WARN]  2024-01-15 14:27:58 payment-service v3.8.2 startup: TransactionCache warm-start enabled "
            "(eviction=disabled) — cache size will grow under high traffic\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:32:01 payment-service java.lang.OutOfMemoryError: Java heap space\n"
            "    TransactionCache.size=937MB - cache eviction is DISABLED\n"
            "[ERROR] 2024-01-15 14:31:58 payment-service OutOfMemoryError (x12 last 60s)\n"
            "[WARN]  2024-01-15 14:31:30 payment-service Memory 94% and rising\n"
            "[WARN]  2024-01-15 14:28:55 payment-service Cache eviction disabled since v3.7.0\n"
            "[INFO]  2024-01-15 14:28:00 payment-service Processing 2,340 txns/min (above average traffic)\n"
            "[WARN]  2024-01-15 14:27:58 payment-service v3.8.2 startup: TransactionCache warm-start enabled "
            "(eviction=disabled) — cache size will grow under high traffic\n"
            "[INFO]  2024-01-15 14:27:55 payment-service Startup OK (v3.8.2)\n"
        ),
    },
    "api-gateway": {
        "all": (
            "[INFO]  2024-01-15 14:32:10 api-gateway All routes nominally healthy — no request errors\n"
            "[WARN]  2024-01-15 14:31:55 api-gateway CPU 68% (health check storms from k8s — expected)\n"
            "[WARN]  2024-01-15 14:31:50 api-gateway p99 latency 1850ms — elevated due to slow upstream: payment-service\n"
            "[INFO]  2024-01-15 14:31:40 api-gateway payment-service route p99 latency: 1850ms (upstream degraded)\n"
            "[INFO]  2024-01-15 14:30:00 api-gateway All other routes nominal. No api-gateway errors.\n"
        ),
        "error": (
            "No error logs for api-gateway.\n"
            "NOTE: CPU 68% and elevated latency (1850ms) are SECONDARY EFFECTS of payment-service degradation,\n"
            "not an api-gateway root cause. Health check storms are expected k8s behavior.\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 14:31:55 api-gateway CPU 68% - health check frequency elevated (k8s)\n"
            "[WARN] 2024-01-15 14:31:50 api-gateway p99 latency 1850ms - upstream payment-service slow\n"
        ),
    },
    "user-service": {
        "all": "[INFO] 2024-01-15 14:32:10 user-service Normal operation. No issues.\n",
        "error": "No errors in user-service.\n",
        "warn": "No warnings in user-service.\n",
    },
}

_RUNBOOKS = {
    "payment-service": (
        "=== RUNBOOK: payment-service ===\n\n"
        "COMMON ISSUES:\n\n"
        "1. HIGH MEMORY / OutOfMemoryError  [MOST COMMON]\n"
        "   Root cause: TransactionCache grows unbounded when cache eviction is disabled\n"
        "               or traffic exceeds the eviction rate.\n"
        "   Detection:\n"
        "     - java.lang.OutOfMemoryError in logs\n"
        "     - MemoryUsage alert > 90%\n"
        "     - TransactionCache.size growing in logs\n"
        "   Fix: Restart the service to clear the heap. The cache is warm-started on boot.\n"
        "     -> execute_remediation(restart, payment-service)\n"
        "   Verify: memory_usage should drop to ~18% within 30 seconds\n"
        "   Follow-up: Check if cache eviction config should be restored.\n\n"
        "2. HIGH LATENCY (without OOM)\n"
        "   Root cause: Downstream DB slowness\n"
        "   Fix: Scale payment-service or investigate DB health.\n\n"
        "3. ELEVATED ERROR RATE (> 5%)\n"
        "   Root cause: Payment gateway credential expiry or downstream timeout\n"
        "   Fix: Check gateway status; rotate credentials if needed.\n\n"
        "ESCALATION: Page #payment-oncall if unresolved after 15 minutes.\n"
    ),
    "oom": (
        "=== RUNBOOK: OutOfMemoryError (Java) ===\n"
        "Quick steps:\n"
        "1. Identify service: check alerts and logs for java.lang.OutOfMemoryError\n"
        "2. Confirm heap exhaustion: GC overhead or TransactionCache.size in logs\n"
        "3. Restart: execute_remediation(restart, <service>)\n"
        "4. Verify: memory_usage drops below 30% post-restart\n"
        "5. Declare resolved with root cause.\n"
    ),
    "memory": (
        "=== RUNBOOK: High Memory Usage ===\n"
        "Java services with memory > 90%:\n"
        "  - Likely cause: unbounded in-memory cache or session store\n"
        "  - Immediate fix: restart the service\n"
        "  - Long-term fix: enable cache eviction / set max cache size\n"
    ),
    "api-gateway": (
        "=== RUNBOOK: api-gateway ===\n"
        "CPU flapping (LOW alert):\n"
        "  - Cause: Kubernetes health check frequency. This is expected behavior.\n"
        "  - Action: No action required. Alert will auto-resolve.\n"
        "  - Do NOT restart api-gateway for this alert.\n"
    ),
}

_DEPLOY_HISTORY = {
    "payment-service": (
        "=== DEPLOY HISTORY: payment-service ===\n"
        "2024-01-15 09:15 | v3.8.2 | alice | ACTIVE  | dependency update (Jackson 2.15.2 -> 2.16.0) — no functional changes\n"
        "2024-01-14 16:40 | v3.8.1 | bob   | RETIRED | bug fix in idempotency key handling\n"
        "2024-01-13 11:00 | v3.8.0 | carol | RETIRED | new payment method support\n"
        "NOTE: Cache eviction was disabled in v3.7.0 (2 weeks ago) as a performance test. "
        "Never re-enabled. This is the underlying configuration issue.\n"
    ),
    "api-gateway": (
        "=== DEPLOY HISTORY: api-gateway ===\n"
        "2024-01-15 11:20 | v5.1.0 | dave  | ACTIVE  | new rate limiting feature — tested in staging, healthy\n"
        "2024-01-14 09:00 | v5.0.9 | eve   | RETIRED | minor config fix\n"
    ),
}

# ---------------------------------------------------------------------------
# IncidentConfig for this task
# ---------------------------------------------------------------------------

OOM_CONFIG = IncidentConfig(
    task_id=1,
    severity_level="SEV2",
    root_cause_service="payment-service",
    root_cause_type="oom",
    root_cause_keywords=["payment-service", "oom", "memory", "heap", "cache", "restart"],
    all_services=["payment-service", "api-gateway", "user-service"],
    relevant_services=["payment-service"],
    red_herring_services=["api-gateway"],
    golden_actions=[("restart", "payment-service")],
    action_order_constraints=[],  # only one action needed
    weights={
        "situational": 0.06,      # reduced by 0.04 (D7 measures investigation quality more precisely)
        "diagnostic": 0.14,       # reduced by 0.06 (D7 captures belief confidence better)
        "remediation": 0.35,
        "time": 0.10,             # reduced by 0.05 (D8 captures ordering quality)
        "communication": 0.10,
        "anti_patterns": 0.10,
        "epistemic_quality": 0.08,
        "workflow_coherence": 0.07,
    },
    hypothesis_space=OOM_HYPOTHESIS_SPACE,
    likelihood_table=OOM_LIKELIHOOD_TABLE,
    true_hypothesis=OOM_TRUE_HYPOTHESIS,
    sla_service="payment-service",
    sla_metric="error_rate",
)


# ---------------------------------------------------------------------------
# OOMIncident
# ---------------------------------------------------------------------------

class OOMIncident(BaseIncident):

    def __init__(self, seed: int = 42):
        engine = RewardEngine(OOM_CONFIG)
        metrics = make_oom_metric_engine()
        belief = BeliefEngine(OOM_HYPOTHESIS_SPACE, OOM_LIKELIHOOD_TABLE, OOM_TRUE_HYPOTHESIS)
        workflow = WorkflowMachine(root_cause_service="payment-service")
        super().__init__(engine, metrics, OOM_CONFIG, belief, workflow, seed)
        self._service_restarted = False

    def get_task_id(self) -> int:
        return 1

    def get_initial_context(self) -> str:
        return (
            "=== INCIDENT OPENED: INC-2024-001 | SEV2 ===\n"
            "Time: 2024-01-15 14:32 UTC\n"
            "Summary: payment-service is degraded. Memory critical. Latency spiking.\n"
            "SLA at risk: payment processing error rate trending upward.\n\n"
            "You are the on-call SRE. Investigate, find the root cause, remediate,\n"
            "and declare resolved with a precise root cause description.\n\n"
            "Tip: Start with list_alerts to see what's firing, then investigate before acting.\n"
        )

    def list_alerts(self) -> tuple[str, float]:
        args = {}
        step_reward = self._tick("list_alerts", args, self.step_count)
        self._record("list_alerts", args, _INITIAL_ALERTS, step_reward)
        return _INITIAL_ALERTS, step_reward

    def query_logs(self, service: str, severity: str = "all", keyword: str = "") -> tuple[str, float]:
        args = {"service": service, "severity": severity, "keyword": keyword}
        self._check_fix_verification("query_logs", args)
        step_reward = self._tick("query_logs", args, self.step_count)

        svc = service.lower().strip()
        sev = (severity or "all").lower().strip()
        logs = _LOGS.get(svc, {})

        if not logs:
            result = f"No logs for '{service}'. Valid: {list(_LOGS.keys())}"
        else:
            raw = logs.get(sev, logs.get("all", ""))
            if keyword:
                lines = [l for l in raw.splitlines() if keyword.lower() in l.lower()]
                result = "\n".join(lines) if lines else f"No logs matching '{keyword}' for {service}."
            else:
                result = raw

        self._record("query_logs", args, result, step_reward)
        return result, step_reward

    def query_metrics(self, service: str, metric: str = "all") -> tuple[str, float]:
        args = {"service": service, "metric": metric}
        self._check_fix_verification("query_metrics", args)
        step_reward = self._tick("query_metrics", args, self.step_count)

        svc = service.lower().strip()
        met = (metric or "all").lower().strip()

        if met == "all":
            result = self.metric_engine.format_all(svc)
        else:
            result = (
                f"=== LIVE METRICS: {svc} ===\n"
                f"  {met}: {self.metric_engine.format_metric(svc, met)}"
            )

        self._record("query_metrics", args, result, step_reward)
        return result, step_reward

    def read_runbook(self, topic: str) -> tuple[str, float]:
        args = {"topic": topic}
        step_reward = self._tick("read_runbook", args, self.step_count)

        topic_lower = topic.lower().strip()
        result = None
        for key, content in _RUNBOOKS.items():
            if key in topic_lower or topic_lower in key:
                result = content
                break
        if result is None:
            result = f"No runbook for '{topic}'. Available: {list(_RUNBOOKS.keys())}"

        self._record("read_runbook", args, result, step_reward)
        return result, step_reward

    def check_deploy_history(self, service: str) -> tuple[str, float]:
        args = {"service": service}
        step_reward = self._tick("check_deploy_history", args, self.step_count)
        svc = service.lower().strip()
        result = _DEPLOY_HISTORY.get(svc, f"No deploy history for '{service}'.")
        self._record("check_deploy_history", args, result, step_reward)
        return result, step_reward

    def execute_remediation(self, action: str, target: str) -> tuple[str, float]:
        args = {"action_type": action, "target": target}
        step_reward = self._tick("execute_remediation", args, self.step_count)

        act = action.lower().strip()
        tgt = target.lower().strip()

        if act == "restart" and "payment" in tgt:
            self._service_restarted = True
            self._after_fix("restart", "payment-service")
            result = (
                "SUCCESS: payment-service restarted.\n"
                "  JVM heap cleared: memory 97% -> 18%\n"
                "  TransactionCache reset (will warm-start)\n"
                "  Latency normalizing... (will take ~30s)\n"
                "  Alerts: MemoryUsage and RequestLatency resolving\n"
                "Recommend: verify metrics, post status, then declare resolved."
            )
        elif act == "restart" and "api-gateway" in tgt:
            result = (
                "SUCCESS: api-gateway restarted. CPU alert unchanged.\n"
                "NOTE: api-gateway CPU alert is caused by k8s health check storms, not a real issue.\n"
                "The payment-service incidents are unaffected."
            )
        elif act == "restart":
            result = f"SUCCESS: {target} restarted. No change to payment-service alerts."
        elif act in ("rollback", "scale", "flush_cache"):
            result = f"Action '{act}' on '{target}' completed. No significant impact on payment-service memory issue."
        else:
            result = f"Unknown action '{act}'. Valid: restart, rollback, scale, flush_cache."

        self._record("execute_remediation", args, result, step_reward)
        return result, step_reward

    def declare_resolved(self, root_cause: str) -> tuple[float, str]:
        args = {"root_cause": root_cause}
        step_reward = self._tick("declare_resolved", args, self.step_count)

        final_score, breakdown = self.reward_engine.compute_final_reward(
            root_cause, self.step_count, self.belief_engine, self.workflow_machine
        )
        feedback = breakdown.to_feedback()
        self.done = True
        self._record("declare_resolved", args, feedback, step_reward)
        return final_score, feedback
