"""
Task 4: Config Drift — TLS Certificate CN Mismatch (Medium-Hard)

Scenario: checkout-service began failing 18 minutes ago with a spike to 18.2% error rate.
The root cause is a TLS certificate rotation on payments-gateway: the cert CN changed from
payments-gw-v1.internal to payments-gw-v2.internal at 09:12 UTC, but checkout-service's
config still references the old CN. All HTTPS calls from checkout-service to payments-gateway
now fail with SSLHandshakeException.

Red herrings:
  1. redis-session — shows elevated connection count (2840) and retries because checkout-service
     sessions fail to commit; looks like a redis issue but it's a downstream symptom.
  2. inventory-service — shows elevated latency and minor error rate because checkout failures
     cause inventory reservations to roll back (downstream cascade), not the root cause.

Key trap: No recent code deploy on checkout-service. Agent must look at infra events
(cert_rotation) in check_deploy_history(payments-gateway), not just code deploys.

Optimal path (8–10 steps):
  list_alerts
  -> query_logs(checkout-service, error)          [SSLHandshakeException CN mismatch]
  -> query_logs(payments-gateway, all)             [cert rotation logged at 09:12]
  -> check_deploy_history(payments-gateway)        [infra event: cert_rotation at 09:12]
  -> read_runbook(tls)                             [confirms update_config fix]
  -> execute_remediation(update_config, checkout-service)
  -> query_metrics(checkout-service)               [verify recovery]
  -> update_status(...)
  -> declare_resolved(...)

IncidentConfig golden actions: [("update_config", "checkout-service")]
"""

from .incident_base import BaseIncident
from .reward_engine import RewardEngine, IncidentConfig
from .metric_engine import make_config_drift_metric_engine
from .belief_engine import (
    BeliefEngine,
    CONFIG_DRIFT_HYPOTHESIS_SPACE,
    CONFIG_DRIFT_TRUE_HYPOTHESIS,
    CONFIG_DRIFT_LIKELIHOOD_TABLE,
)
from .workflow_machine import WorkflowMachine

_INITIAL_ALERTS = """
=== FIRING ALERTS ===
[HIGH]   checkout-service  | ErrorRate     | 18.2%    | threshold: 5%    | duration: 18m
[HIGH]   checkout-service  | Latency_P99   | 3120ms   | threshold: 500ms | duration: 18m
[MEDIUM] checkout-service  | RequestVolume | 920/min (down 34%) | threshold: -25% | duration: 15m
[MEDIUM] redis-session      | ConnectionCount | 2840   | threshold: 2000  | duration: 12m
[LOW]    inventory-service  | ErrorRate     | 3.1%     | threshold: 2%    | duration: 10m
"""

_LOGS = {
    "checkout-service": {
        "error": (
            "[ERROR] 2024-01-15 09:30:44 checkout-service SSLHandshakeException: No subject alternative names present\n"
            "    at sun.security.ssl.Handshaker.throwSSLException(Handshaker.java:1439)\n"
            "    at sun.security.ssl.ClientHandshaker.serverCertificate(ClientHandshaker.java:1587)\n"
            "    Caused by: javax.net.ssl.SSLHandshakeException: No subject alternative names present\n"
            "    Remote host: payments-gateway:8443 | Expected CN: payments-gw-v1.internal\n"
            "    Actual CN: payments-gw-v2.internal — CN MISMATCH\n"
            "[ERROR] 2024-01-15 09:30:41 checkout-service Failed to POST /v1/payments/charge — SSL error (x84 last 2min)\n"
            "[ERROR] 2024-01-15 09:29:55 checkout-service Connection to payments-gateway rejected by TLS handshake\n"
            "[ERROR] 2024-01-15 09:12:31 checkout-service First SSL failure detected (onset matches payments-gateway event)\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 09:30:43 checkout-service Retrying payments-gateway call (attempt 3/3) — still failing\n"
            "[WARN] 2024-01-15 09:30:40 checkout-service Circuit breaker HALF-OPEN: payments-gateway endpoint\n"
            "[WARN] 2024-01-15 09:28:14 checkout-service Session commit failures increasing: redis-session retry storms\n"
            "[WARN] 2024-01-15 09:12:45 checkout-service TLS verification failing for payments-gateway (first occurrence)\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 09:30:44 checkout-service SSLHandshakeException: CN mismatch payments-gw-v1.internal != payments-gw-v2.internal\n"
            "[ERROR] 2024-01-15 09:30:41 checkout-service POST /v1/payments/charge SSL error (x84 last 2min)\n"
            "[WARN]  2024-01-15 09:30:43 checkout-service Retrying payments-gateway (attempt 3/3)\n"
            "[WARN]  2024-01-15 09:30:40 checkout-service Circuit breaker HALF-OPEN: payments-gateway\n"
            "[WARN]  2024-01-15 09:28:14 checkout-service redis-session retry storms (downstream from checkout failures)\n"
            "[INFO]  2024-01-15 09:12:30 checkout-service Startup healthy — config references payments-gw-v1.internal (tls_cn)\n"
            "[INFO]  2024-01-15 09:11:50 checkout-service All health checks PASSING pre-09:12\n"
            "[INFO]  2024-01-15 09:10:00 checkout-service Last deploy: v4.1.2 at 08:55 UTC (19 minutes before SSL failure)\n"
        ),
    },
    "payments-gateway": {
        "error": (
            "No errors in payments-gateway. Service is operating normally.\n"
            "NOTE: payments-gateway is HEALTHY — it is not the failing service.\n"
            "      The TLS cert rotation it performed is CORRECT infra procedure.\n"
            "      The misconfiguration is in checkout-service's expected CN.\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 09:12:04 payments-gateway TLS certificate rotated successfully\n"
            "  Old CN: payments-gw-v1.internal | New CN: payments-gw-v2.internal\n"
            "  Cert expiry extended from 2024-02-01 to 2024-08-01\n"
            "[WARN] 2024-01-15 09:12:02 payments-gateway Cert rotation initiated by infra-team (scheduled maintenance)\n"
        ),
        "all": (
            "[INFO]  2024-01-15 09:30:44 payments-gateway Processing requests normally: 1,240/min\n"
            "[INFO]  2024-01-15 09:30:40 payments-gateway Error rate: 0.01% (healthy)\n"
            "[WARN]  2024-01-15 09:12:04 payments-gateway TLS cert rotated: CN payments-gw-v1.internal -> payments-gw-v2.internal\n"
            "[WARN]  2024-01-15 09:12:02 payments-gateway Cert rotation by infra-team (scheduled). New cert valid until 2024-08-01.\n"
            "[INFO]  2024-01-15 09:11:58 payments-gateway Pre-rotation status: all healthy\n"
            "[INFO]  2024-01-15 08:55:00 payments-gateway Receiving checkout-service payments normally (pre-rotation)\n"
        ),
    },
    "redis-session": {
        "error": (
            "[ERROR] 2024-01-15 09:30:42 redis-session Session commit timeout: checkout-session-8f3a (attempt 3/3)\n"
            "[ERROR] 2024-01-15 09:30:38 redis-session Session commit timeout: checkout-session-9c1b (attempt 3/3)\n"
            "NOTE: These timeouts are CAUSED BY checkout-service failures — sessions cannot commit\n"
            "      because checkout payment calls never complete. This is a symptom, not a cause.\n"
            "      redis-session memory usage and latency are within normal bounds.\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 09:30:40 redis-session High connection count: 2840/3000 (checkout retry storms)\n"
            "[WARN] 2024-01-15 09:29:15 redis-session Connection pool pressure: checkout-service holding connections during SSL retries\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 09:30:42 redis-session Session commit timeout (checkout retries exhausting connections)\n"
            "[WARN]  2024-01-15 09:30:40 redis-session Connection count 2840/3000 — upstream checkout retries\n"
            "[INFO]  2024-01-15 09:30:35 redis-session Serving 12,440 active sessions (auth-service, user-service, checkout-service)\n"
            "[INFO]  2024-01-15 09:30:30 redis-session Memory usage: 68% — normal. Latency p99: 4ms — normal.\n"
            "[INFO]  2024-01-15 09:12:00 redis-session All healthy pre-09:12 (connection count was 890)\n"
        ),
    },
    "inventory-service": {
        "error": (
            "[ERROR] 2024-01-15 09:30:38 inventory-service ReservationRollback: checkout payment failed, releasing reservation\n"
            "[ERROR] 2024-01-15 09:30:32 inventory-service ReservationRollback (x41 last 2min) — upstream checkout failures\n"
            "NOTE: inventory-service errors are downstream rollbacks caused by checkout payment failures.\n"
            "      inventory-service itself is healthy. Fix checkout-service to resolve.\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 09:30:35 inventory-service Elevated rollback rate: 41 rollbacks/2min (up from baseline 2/2min)\n"
            "[WARN] 2024-01-15 09:29:00 inventory-service Reservation churn increasing — monitor\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 09:30:38 inventory-service ReservationRollback: checkout payment failed (x41 last 2min)\n"
            "[WARN]  2024-01-15 09:30:35 inventory-service Elevated rollback rate (downstream symptom of checkout failures)\n"
            "[INFO]  2024-01-15 09:30:30 inventory-service Health check: PASSING. Error rate 3.1% — all are rollbacks.\n"
            "[INFO]  2024-01-15 09:30:25 inventory-service Core inventory operations: 99.7% success rate\n"
            "[INFO]  2024-01-15 09:12:00 inventory-service Pre-incident: rollback rate 0.4/min (baseline)\n"
        ),
    },
}

_RUNBOOKS = {
    "tls": (
        "=== RUNBOOK: TLS Certificate Issues ===\n\n"
        "SYMPTOM: SSLHandshakeException or TLS verification failures between services.\n\n"
        "COMMON CAUSES:\n"
        "  1. CN mismatch: Certificate Common Name (CN) changed but client config not updated.\n"
        "     Detection: SSLHandshakeException with 'No subject alternative names' or 'CN mismatch'.\n"
        "     Fix: Update the client service config to reference the new CN.\n"
        "       -> execute_remediation(update_config, <client-service>)\n\n"
        "  2. Certificate expired.\n"
        "     Detection: SSLHandshakeException with 'certificate expired'.\n"
        "     Fix: Rotate cert on the server side, update client trust store.\n\n"
        "  3. Trust store not updated after cert rotation.\n"
        "     Detection: 'unable to find valid certification path'.\n"
        "     Fix: Restart the client service to reload trust store.\n\n"
        "DIAGNOSTIC STEPS:\n"
        "  1. Check client service logs for exact SSL error message.\n"
        "  2. Check server (payments-gateway) logs for recent cert rotations.\n"
        "  3. Compare expected CN in client config with actual CN in server cert.\n"
        "  4. Update client config if CN mismatch found.\n\n"
        "NOTE: Restarting the CLIENT service will NOT fix a CN mismatch — the config\n"
        "      still references the old CN after restart. Use update_config instead.\n\n"
        "ESCALATION: Page #platform-security if cert rotation was unscheduled.\n"
    ),
    "checkout-service": (
        "=== RUNBOOK: checkout-service ===\n\n"
        "ELEVATED ERROR RATE:\n"
        "  1. Check for recent deploys: check_deploy_history(checkout-service)\n"
        "  2. Check for upstream dependency failures (payments-gateway, inventory-service)\n"
        "  3. If SSL errors: see TLS runbook. Fix is update_config, NOT restart.\n\n"
        "CONFIGURATION:\n"
        "  checkout-service config includes: payments_gateway_tls_cn (Common Name for TLS verification)\n"
        "  If this CN is stale after a cert rotation, update via:\n"
        "    -> execute_remediation(update_config, checkout-service)\n\n"
        "NOTE: A restart of checkout-service will NOT resolve CN mismatch issues — the\n"
        "      config file is reloaded on restart but the CN mismatch persists.\n"
        "      Only update_config pushes a new config with the correct CN.\n"
    ),
    "payments-gateway": (
        "=== RUNBOOK: payments-gateway ===\n\n"
        "SCHEDULED MAINTENANCE: TLS cert rotation is a routine infra procedure.\n"
        "  Rotation schedule: quarterly, performed by infra-team.\n"
        "  After rotation: client services MUST update their tls_cn config reference.\n"
        "  Affected clients: checkout-service, order-service.\n\n"
        "IF payments-gateway appears healthy but clients are failing:\n"
        "  -> The client has a stale CN config. See TLS runbook.\n"
        "  -> Do NOT restart payments-gateway — it will not fix the client config.\n\n"
        "ESCALATION: Page #infra-team if unexpected cert rotation detected.\n"
    ),
    "redis-session": (
        "=== RUNBOOK: redis-session ===\n\n"
        "HIGH CONNECTION COUNT:\n"
        "  Threshold: 3000 connections.\n"
        "  Common cause: Upstream service retry storms holding connections open.\n"
        "  IMPORTANT: High connection count is USUALLY a SYMPTOM of an upstream issue.\n"
        "              Investigate upstream services (checkout-service, auth-service) first.\n\n"
        "  If upstream services are healthy and redis-session connection count is still high:\n"
        "    -> execute_remediation(flush_cache, redis-session) to clear stale connections\n"
        "    -> execute_remediation(restart, redis-session) as last resort\n\n"
        "NOTE: In this incident, redis-session elevated connections are caused by\n"
        "      checkout-service retry storms. Fix checkout-service first.\n"
    ),
    "inventory-service": (
        "=== RUNBOOK: inventory-service ===\n\n"
        "ELEVATED ROLLBACK RATE:\n"
        "  Rollbacks occur when upstream services (checkout-service) fail mid-transaction.\n"
        "  If checkout-service is failing, expect rollback noise in inventory-service.\n"
        "  This is expected behavior — no action needed on inventory-service itself.\n\n"
        "  Fix the upstream failure (checkout-service) to resolve rollback noise.\n"
    ),
}

_DEPLOY_HISTORY = {
    "checkout-service": (
        "=== DEPLOY HISTORY: checkout-service ===\n"
        "2024-01-15 08:55 | v4.1.2 | alice | ACTIVE  | minor UI text fix on checkout summary page (no logic changes)\n"
        "2024-01-14 16:20 | v4.1.1 | alice | RETIRED | form validation improvements\n"
        "2024-01-13 11:00 | v4.1.0 | bob   | RETIRED | payment flow refactor\n"
        "\nNOTE: v4.1.2 deployed at 08:55 UTC — 17 minutes BEFORE error onset at 09:12.\n"
        "      The deploy had no payment or TLS logic changes. No correlation with error spike.\n"
    ),
    "payments-gateway": (
        "=== DEPLOY HISTORY: payments-gateway ===\n"
        "2024-01-15 09:12 | INFRA  | infra-team | cert_rotation | TLS certificate rotated (scheduled quarterly maintenance)\n"
        "                   CN change: payments-gw-v1.internal -> payments-gw-v2.internal\n"
        "                   Cert expiry extended: 2024-02-01 -> 2024-08-01\n"
        "2024-01-10 14:00 | v5.2.1 | dave        | ACTIVE        | payment processing performance improvement\n"
        "2024-01-05 09:30 | v5.2.0 | dave        | RETIRED       | fraud detection rule update\n"
        "\n*** INFRA EVENT at 09:12: cert_rotation — client services using old CN will fail ***\n"
        "NOTE: Error onset in checkout-service at 09:12 exactly matches cert rotation timestamp.\n"
    ),
    "redis-session": (
        "=== DEPLOY HISTORY: redis-session ===\n"
        "2023-12-20 10:00 | v3.1.4 | ops-team | ACTIVE | patch for CVE-2023-xxxx (26 days ago, stable)\n"
    ),
    "inventory-service": (
        "=== DEPLOY HISTORY: inventory-service ===\n"
        "2024-01-14 13:00 | v2.4.0 | carol | ACTIVE | stock level API improvements (yesterday, stable)\n"
    ),
}

CONFIG_DRIFT_CONFIG = IncidentConfig(
    task_id=4,
    severity_level="SEV2",
    root_cause_service="payments-gateway",
    root_cause_type="config_drift",
    root_cause_keywords=["tls", "cn", "certificate", "ssl", "config", "payments-gateway", "checkout-service"],
    all_services=["checkout-service", "payments-gateway", "redis-session", "inventory-service"],
    relevant_services=["checkout-service", "payments-gateway"],
    red_herring_services=["redis-session", "inventory-service"],
    golden_actions=[("update_config", "checkout-service")],
    action_order_constraints=[],
    weights={
        "situational":       0.08,
        "diagnostic":        0.19,
        "remediation":       0.28,
        "time":              0.10,
        "communication":     0.10,
        "anti_patterns":     0.10,
        "epistemic_quality": 0.08,
        "workflow_coherence": 0.07,
    },
    hypothesis_space=CONFIG_DRIFT_HYPOTHESIS_SPACE,
    likelihood_table=CONFIG_DRIFT_LIKELIHOOD_TABLE,
    true_hypothesis=CONFIG_DRIFT_TRUE_HYPOTHESIS,
    sla_service="checkout-service",
    sla_metric="error_rate",
)


class ConfigDriftIncident(BaseIncident):

    def __init__(self, seed: int = 42):
        engine = RewardEngine(CONFIG_DRIFT_CONFIG)
        metrics = make_config_drift_metric_engine()
        belief = BeliefEngine(
            CONFIG_DRIFT_HYPOTHESIS_SPACE,
            CONFIG_DRIFT_LIKELIHOOD_TABLE,
            CONFIG_DRIFT_TRUE_HYPOTHESIS,
        )
        workflow = WorkflowMachine(root_cause_service="payments-gateway")
        super().__init__(engine, metrics, CONFIG_DRIFT_CONFIG, belief, workflow, seed)
        self._config_updated = False
        self._restarted_checkout = False
        self._restarted_payments = False
        self._flushed_redis = False

    def get_task_id(self) -> int:
        return 4

    def get_initial_context(self) -> str:
        return (
            "=== INCIDENT OPENED: INC-2024-004 | SEV2 ===\n"
            "Time: 2024-01-15 09:30 UTC\n"
            "Summary: checkout-service error rate has been elevated for 18 minutes.\n"
            "Customers are unable to complete purchases. Revenue impact ongoing.\n\n"
            "Initial triage notes: No recent code deploy on checkout-service coincides\n"
            "with the error onset. The errors began at 09:12 UTC. Investigate what\n"
            "changed in the infrastructure around that time.\n\n"
            "Tip: Infrastructure changes (cert rotations, config updates, infra events)\n"
            "do not always appear as code deploys. Check deploy history carefully.\n"
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
            result = f"=== LIVE METRICS: {svc} ===\n  {met}: {self.metric_engine.format_metric(svc, met)}"

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

        if act == "update_config" and "checkout" in tgt:
            self._config_updated = True
            self._after_fix("update_config", "checkout-service")
            result = (
                "SUCCESS: checkout-service config updated.\n"
                "  payments_gateway_tls_cn: payments-gw-v1.internal -> payments-gw-v2.internal\n"
                "  Config reloaded without restart. TLS handshake now succeeds.\n"
                "  SSLHandshakeExceptions stopped immediately.\n"
                "  Error rate recovering: 18.2% -> 0.5% (estimated <1 min to full recovery)\n"
                "  Circuit breaker resetting to CLOSED state.\n"
                "  redis-session connection storm resolving (retry pressure removed).\n"
                "  inventory-service rollback rate normalizing.\n"
                "Recommend: verify metrics, then declare resolved."
            )
        elif act == "restart" and "checkout" in tgt:
            self._restarted_checkout = True
            self._after_fix("restart", "checkout-service")
            result = (
                "checkout-service restarted.\n"
                "  Service back online in 8 seconds.\n"
                "  PARTIAL IMPROVEMENT: error_rate 18.2% -> 11.4% (briefly, during warmup).\n"
                "  REGRESSION: error rate climbing back: 11.4% -> 14.8% -> 17.1%\n"
                "  NOTICE: SSLHandshakeException errors RESUMING — CN mismatch persists in config.\n"
                "  Restart reloaded config but config still references old CN: payments-gw-v1.internal\n"
                "  HINT: A config update (not restart) is needed to push the corrected CN value.\n"
                "  Root cause still active."
            )
        elif act == "restart" and "payments" in tgt:
            self._restarted_payments = True
            result = (
                "payments-gateway restarted.\n"
                "  Service back online with same TLS certificate (CN: payments-gw-v2.internal).\n"
                "  checkout-service error rate: UNCHANGED at 18.2%.\n"
                "  NOTICE: payments-gateway was healthy — the issue is the CLIENT config,\n"
                "          not the gateway itself. Restarting the gateway has no effect.\n"
                "  Root cause still active."
            )
        elif act == "flush_cache" and "redis" in tgt:
            self._flushed_redis = True
            result = (
                "redis-session cache flushed.\n"
                "  Connection count temporarily drops: 2840 -> 340.\n"
                "  NOTICE: checkout-service error rate UNCHANGED at 18.2%.\n"
                "  Within 30 seconds, redis-session connections climbing again: 340 -> 1840\n"
                "  (checkout retry storms are refilling the connection pool).\n"
                "  NOTICE: redis-session was a symptom, not the cause.\n"
                "  Fix checkout-service to resolve the connection pressure.\n"
                "  Root cause still active."
            )
        elif act == "scale" and "checkout" in tgt:
            result = (
                "checkout-service scaled to 8 pods.\n"
                "  Latency P99: 3120ms -> 2840ms (marginal improvement from extra capacity).\n"
                "  Error rate: UNCHANGED at 18.2% — SSL errors are not a capacity problem.\n"
                "  Root cause still active."
            )
        else:
            result = f"Action '{act}' on '{target}': completed. No significant change to checkout-service alerts."

        self._record("execute_remediation", args, result, step_reward)
        return result, step_reward

    def declare_resolved(self, root_cause: str) -> tuple[float, str]:
        args = {"root_cause": root_cause}
        step_reward = self._tick("declare_resolved", args, self.step_count)

        # Extra penalty for wrong remediations
        if self._restarted_payments:
            self.reward_engine.wrong_actions_count += 1
        if self._flushed_redis:
            self.reward_engine.wrong_actions_count += 1

        final_score, breakdown = self.reward_engine.compute_final_reward(
            root_cause, self.step_count, self.belief_engine, self.workflow_machine
        )
        feedback = breakdown.to_feedback()
        self.done = True
        self._record("declare_resolved", args, feedback, step_reward)
        return final_score, feedback
