"""
Task 2: Bad Deploy Rollback (Medium)

Scenario: api-gateway v2.3.1 was deployed 5 minutes ago. It removed lazy
initialization from RouteHandler, causing NullPointerException on every request
that hits a route not yet cached. Error rate spiked to 12.4%.

Red herrings:
  1. user-service v1.5.0 deployed 44 mins ago — looks suspicious but is fine
  2. web-frontend CPU at 71% — unrelated marketing campaign traffic, flapping

Key trap: two recent deploys exist. Agent must correlate the error spike TIMING
with api-gateway deploy (at 15:42), not user-service deploy (at 15:03).

Optimal path (6–9 steps):
  list_alerts -> query_logs(api-gateway, error) -> check_deploy_history(api-gateway)
  -> query_metrics(api-gateway, error_rate) [see timing correlation]
  -> execute_remediation(rollback, api-gateway)
  -> query_metrics(api-gateway, error_rate) [verify recovery]
  -> update_status(...) -> declare_resolved(...)

IncidentConfig golden actions: [("rollback", "api-gateway")]
"""

from .incident_base import BaseIncident
from .reward_engine import RewardEngine, IncidentConfig
from .metric_engine import make_deploy_metric_engine

_INITIAL_ALERTS = """
=== FIRING ALERTS ===
[HIGH]   api-gateway   | ErrorRate     | 12.4%    | threshold: 5%    | duration: 5m
[HIGH]   api-gateway   | Latency_P99   | 2350ms   | threshold: 500ms | duration: 5m
[MEDIUM] api-gateway   | RequestVolume | 1840/min (down 28%) | threshold: -20% | duration: 4m
[LOW]    web-frontend  | CPUUsage      | 71%      | threshold: 70%   | duration: 8m  | flapping
"""

_LOGS = {
    "api-gateway": {
        "error": (
            "[ERROR] 2024-01-15 15:47:12 api-gateway NullPointerException in RouteHandler\n"
            "    at com.gateway.routing.RouteHandler.resolve(RouteHandler.java:203)\n"
            "    at com.gateway.filter.AuthFilter.doFilter(AuthFilter.java:88)\n"
            "    Caused by: java.lang.NullPointerException: routingTable is null\n"
            "    NOTE: routingTable lazy init was REMOVED in v2.3.1 - never populated on startup\n"
            "[ERROR] 2024-01-15 15:47:08 api-gateway NullPointerException RouteHandler (x47 last 30s)\n"
            "[ERROR] 2024-01-15 15:46:59 api-gateway NullPointerException RouteHandler (x23 last 30s)\n"
            "[ERROR] 2024-01-15 15:46:41 api-gateway NPE rate correlates exactly with deploy at 15:42\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 15:47:14 api-gateway Circuit breaker OPEN: /api/payments\n"
            "[WARN] 2024-01-15 15:47:10 api-gateway Circuit breaker OPEN: /api/users\n"
            "[WARN] 2024-01-15 15:46:42 api-gateway Route resolution failures increasing\n"
            "[WARN] 2024-01-15 15:43:01 api-gateway First NPE detected (1 min after v2.3.1 deploy)\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 15:47:12 api-gateway NullPointerException: routingTable is null (v2.3.1)\n"
            "[ERROR] 2024-01-15 15:47:08 api-gateway NullPointerException (x47 last 30s)\n"
            "[WARN]  2024-01-15 15:47:14 api-gateway Circuit breaker OPEN: /api/payments\n"
            "[WARN]  2024-01-15 15:47:10 api-gateway Circuit breaker OPEN: /api/users\n"
            "[INFO]  2024-01-15 15:42:55 api-gateway Deployment v2.3.1 applied successfully\n"
            "[INFO]  2024-01-15 15:42:50 api-gateway Startup complete (v2.3.1)\n"
            "[INFO]  2024-01-15 15:42:00 api-gateway Pre-deploy traffic: error_rate 0.02% (healthy)\n"
        ),
    },
    "user-service": {
        "error": "No errors in user-service. v1.5.0 is healthy — only read-only cache changes.\n",
        "warn": "No warnings in user-service.\n",
        "all": (
            "[INFO] 2024-01-15 15:47:15 user-service 1,204 profile requests processed (normal)\n"
            "[INFO] 2024-01-15 15:47:10 user-service All health checks passing\n"
            "[INFO] 2024-01-15 15:47:05 user-service Cache hit rate: 94% (healthy)\n"
            "[INFO] 2024-01-15 15:03:22 user-service Deployment v1.5.0 applied successfully\n"
            "[INFO] 2024-01-15 15:03:18 user-service Startup complete (v1.5.0) - no issues\n"
        ),
    },
    "payment-service": {
        "error": "No errors in payment-service. Circuit breaker from api-gateway is throttling upstream requests.\n",
        "all": (
            "[INFO] 2024-01-15 15:47:15 payment-service Healthy - fewer requests due to upstream circuit breaker\n"
            "[WARN] 2024-01-15 15:47:10 payment-service Reduced traffic from api-gateway (circuit breaker open)\n"
        ),
        "warn": "[WARN] 2024-01-15 15:47:10 payment-service Reduced incoming requests - upstream issue\n",
    },
    "web-frontend": {
        "all": (
            "[INFO] 2024-01-15 15:47:15 web-frontend 2,890 req/sec - marketing campaign driving traffic\n"
            "[INFO] 2024-01-15 15:47:10 web-frontend CPU high but stable: serving cached content\n"
            "[INFO] 2024-01-15 15:46:00 web-frontend Campaign 'winter-sale' started at 15:30 (expected traffic spike)\n"
        ),
        "error": "No errors in web-frontend. CPU alert is expected due to marketing campaign.\n",
        "warn": "[WARN] 2024-01-15 15:47:00 web-frontend CPU 71% (expected: marketing campaign active)\n",
    },
}

_RUNBOOKS = {
    "api-gateway": (
        "=== RUNBOOK: api-gateway ===\n\n"
        "1. ELEVATED ERROR RATE AFTER DEPLOY\n"
        "   Root cause: Bad deployment introduced a regression.\n"
        "   Detection: ErrorRate spike timestamp correlates with deploy timestamp.\n"
        "   Fix: Roll back to the previous version.\n"
        "     -> check_deploy_history(api-gateway) - identify the bad deploy\n"
        "     -> execute_remediation(rollback, api-gateway)\n"
        "   Verify: error_rate should drop below 1% within 2 minutes post-rollback.\n\n"
        "2. HIGH LATENCY (no error spike)\n"
        "   Fix: execute_remediation(scale, api-gateway)\n\n"
        "3. CPU ALERT (LOW severity, flapping)\n"
        "   Usually caused by k8s health check storms. No action needed.\n\n"
        "ESCALATION: Page #platform-oncall if error_rate > 20% or not resolved in 10 min.\n"
    ),
    "rollback": (
        "=== RUNBOOK: Service Rollback ===\n"
        "1. Identify the service with the regression.\n"
        "2. Confirm: error spike timing matches deploy timestamp (use query_metrics for timeline).\n"
        "3. Confirm: error logs reference new version code (e.g., line numbers, version strings).\n"
        "4. Execute: execute_remediation(rollback, <service>)\n"
        "5. Verify: query_metrics to confirm error_rate recovering.\n"
        "6. Post status update. Declare resolved.\n"
    ),
    "user-service": (
        "=== RUNBOOK: user-service ===\n"
        "v1.5.0 changelog (deployed 15:03 today):\n"
        "  - Profile cache performance improvements\n"
        "  - Read-only changes to cache layer\n"
        "  - No routing, auth, or request handling changes\n"
        "NOTE: v1.5.0 is NOT the cause of api-gateway errors (different service, different layer).\n"
    ),
    "web-frontend": (
        "=== RUNBOOK: web-frontend ===\n"
        "HIGH CPU:\n"
        "  - Marketing campaigns cause expected traffic spikes.\n"
        "  - Check with #marketing-team if a campaign is running.\n"
        "  - No action required unless CPU exceeds 90% for > 15 minutes.\n"
        "NOTE: web-frontend CPU is UNRELATED to api-gateway error rate.\n"
    ),
}

_DEPLOY_HISTORY = {
    "api-gateway": (
        "=== DEPLOY HISTORY: api-gateway ===\n"
        "2024-01-15 15:42 | v2.3.1 | dave  | ACTIVE  | CHANGE: removed lazy init from RouteHandler for 'performance' - routingTable now must be pre-populated\n"
        "2024-01-15 09:10 | v2.3.0 | eve   | RETIRED | new auth middleware (healthy for 6+ hours)\n"
        "2024-01-14 14:30 | v2.2.9 | frank | RETIRED | performance tuning\n"
        "\nNOTE: v2.3.1 deployed at 15:42. Error rate spike began at 15:43. Correlation: 1 minute.\n"
    ),
    "user-service": (
        "=== DEPLOY HISTORY: user-service ===\n"
        "2024-01-15 15:03 | v1.5.0 | grace | ACTIVE  | profile cache performance improvements (read-only, no API changes)\n"
        "2024-01-14 11:20 | v1.4.9 | henry | RETIRED | password reset bug fix\n"
        "\nNOTE: v1.5.0 deployed at 15:03 - 44 minutes before error spike. No correlation.\n"
    ),
    "payment-service": (
        "=== DEPLOY HISTORY: payment-service ===\n"
        "2024-01-14 16:40 | v3.8.1 | bob | ACTIVE | no recent changes (deployed yesterday)\n"
    ),
    "web-frontend": (
        "=== DEPLOY HISTORY: web-frontend ===\n"
        "2024-01-13 10:00 | v8.2.0 | carol | ACTIVE | 2 days ago - stable\n"
    ),
}

DEPLOY_CONFIG = IncidentConfig(
    task_id=2,
    severity_level="SEV2",
    root_cause_service="api-gateway",
    root_cause_type="bad_deploy",
    root_cause_keywords=["api-gateway", "v2.3.1", "rollback", "deploy", "nullpointer", "routing"],
    all_services=["api-gateway", "user-service", "payment-service", "web-frontend"],
    relevant_services=["api-gateway", "user-service"],
    red_herring_services=["web-frontend", "user-service"],
    golden_actions=[("rollback", "api-gateway")],
    action_order_constraints=[],
    weights={
        "situational": 0.12,
        "diagnostic": 0.25,
        "remediation": 0.28,
        "time": 0.15,
        "communication": 0.10,
        "anti_patterns": 0.10,
    },
    sla_service="api-gateway",
    sla_metric="error_rate",
)


class DeployIncident(BaseIncident):

    def __init__(self, seed: int = 42):
        engine = RewardEngine(DEPLOY_CONFIG)
        metrics = make_deploy_metric_engine()
        super().__init__(engine, metrics, DEPLOY_CONFIG, seed)
        self._rolled_back_gateway = False
        self._rolled_back_user_service = False

    def get_task_id(self) -> int:
        return 2

    def get_initial_context(self) -> str:
        return (
            "=== INCIDENT OPENED: INC-2024-002 | SEV2 ===\n"
            "Time: 2024-01-15 15:47 UTC\n"
            "Summary: api-gateway degraded. Error rate spiking, latency elevated.\n"
            "Multiple services may be affected. SLA at risk.\n\n"
            "Clue: Two deployments happened in the last hour. One may be the cause.\n"
            "Investigate the error pattern before rolling back — rolling back the wrong\n"
            "service wastes time and causes unnecessary disruption.\n\n"
            "Tip: Check logs for error details, then deploy history to find the culprit.\n"
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

        if act == "rollback" and "api-gateway" in tgt:
            self._rolled_back_gateway = True
            self._after_fix("rollback", "api-gateway")
            result = (
                "SUCCESS: api-gateway rolled back to v2.3.0.\n"
                "  RouteHandler lazy initialization restored.\n"
                "  NullPointerExceptions stopped immediately.\n"
                "  Error rate recovering: 12.4% -> 1.2% (estimated 2 min to full recovery)\n"
                "  Circuit breakers resetting.\n"
                "Recommend: verify metrics in 2 minutes, then declare resolved."
            )
        elif act == "rollback" and "user-service" in tgt:
            self._rolled_back_user_service = True
            result = (
                "user-service rolled back to v1.4.9.\n"
                "NOTICE: api-gateway alerts still firing. user-service was NOT the cause.\n"
                "The v1.5.0 deploy had no routing/request handling changes."
            )
        elif act == "scale" and "api-gateway" in tgt:
            result = (
                "api-gateway scaled to 6 pods. Partial improvement: latency 2350ms -> 1800ms.\n"
                "Error rate unchanged at 12.4%. The bug is in the code, not capacity."
            )
        elif act == "rollback" and "web-frontend" in tgt:
            result = "web-frontend rollback completed. api-gateway alerts unchanged. web-frontend was unrelated."
        else:
            result = f"Action '{act}' on '{target}': completed. No significant change to api-gateway alerts."

        self._record("execute_remediation", args, result, step_reward)
        return result, step_reward

    def declare_resolved(self, root_cause: str) -> tuple[float, str]:
        args = {"root_cause": root_cause}
        step_reward = self._tick("declare_resolved", args, self.step_count)

        # Extra penalty: rolled back wrong service
        if self._rolled_back_user_service:
            self.reward_engine.wrong_actions_count += 1

        final_score, breakdown = self.reward_engine.compute_final_reward(
            root_cause, self.step_count
        )
        feedback = breakdown.to_feedback()
        self.done = True
        self._record("declare_resolved", args, feedback, step_reward)
        return final_score, feedback
