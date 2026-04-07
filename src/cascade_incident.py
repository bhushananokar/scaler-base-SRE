"""
Task 3: Cascading Infrastructure Failure (Hard)

Scenario: order-service v4.2.0 introduced async query batching that acquires DB
connections without releasing them (connection leak). This exhausts db-pool
(200/200 connections, 847+ waiting). All DB-dependent services fail.

Time bomb: db-pool wait_queue grows ~80 connections/step. At 2000+, recovery
takes significantly longer. The SLA clock ticks aggressively.

Red herrings:
  1. web-frontend CPU at 78% — marketing campaign, completely unrelated
  2. payment-service v3.8.2 deployed 2 days ago — red herring; healthy

The trap: restarting downstream services (payment, inventory, shipping) gives
temporary relief (they reconnect briefly) but fail again within 2 steps as
the pool re-exhausts. Only fixing order-service + scaling db-pool resolves it.

Ordering constraint: restart/rollback order-service BEFORE scale db-pool
achieves maximum score (stop the leak before clearing the queue). But both
actions are acceptable in any order.

Optimal path (8–12 steps):
  list_alerts
  -> query_logs(payment-service, error)     [see "pool exhausted"]
  -> query_logs(db-pool, error)             [confirm pool exhausted]
  -> query_metrics(db-pool, all)            [see connections_used: 200/200, growth since 16:04]
  -> query_metrics(order-service, db_connections_acquired)  [find the leak source]
  -> check_deploy_history(order-service)    [confirm v4.2.0 at 16:03]
  -> update_status("Root cause: order-service connection leak. Fixing.")
  -> execute_remediation(restart, order-service)
  -> execute_remediation(scale, db-pool)
  -> query_metrics(payment-service, error_rate)  [verify recovery]
  -> declare_resolved(...)

IncidentConfig golden actions: [("restart", "order-service"), ("scale", "db-pool")]
Ordering constraint: restart order-service (idx 0) before scale db-pool (idx 1)
"""

from .incident_base import BaseIncident
from .reward_engine import RewardEngine, IncidentConfig
from .metric_engine import make_cascade_metric_engine

_DB_TIMEOUT = (
    "DB connection timeout: could not acquire connection from pool\n"
    "  pool_size=200, active=200, waiting=847+\n"
    "  HikariPool$PoolTimeoutException: Connection not available after 30000ms"
)

_INITIAL_ALERTS = """
=== FIRING ALERTS (8 active) ===
[CRITICAL] payment-service   | ErrorRate       | 89.2% | threshold: 5%    | duration: 12m | RISING
[CRITICAL] inventory-service | ErrorRate       | 91.4% | threshold: 5%    | duration: 11m | RISING
[CRITICAL] shipping-service  | ErrorRate       | 87.8% | threshold: 5%    | duration: 10m | RISING
[HIGH]     db-pool           | ConnectionUsage | 100%  | threshold: 80%   | duration: 14m | EXHAUSTED
[HIGH]     db-pool           | WaitQueueDepth  | 847   | threshold: 100   | duration: 13m | GROWING
[HIGH]     payment-service   | Latency_P99     | 30000ms (timeout)        | duration: 12m
[HIGH]     inventory-service | Latency_P99     | 30000ms (timeout)        | duration: 11m
[MEDIUM]   order-service     | DBConnectionRate| 340/min | threshold: 50/min | duration: 15m
[LOW]      web-frontend      | CPUUsage        | 78%   | threshold: 70%   | duration: 20m | flapping
"""

_LOGS = {
    "payment-service": {
        "error": (
            f"[ERROR] 2024-01-15 16:18:44 payment-service {_DB_TIMEOUT}\n"
            f"[ERROR] 2024-01-15 16:18:43 payment-service {_DB_TIMEOUT} (x89 last 60s)\n"
            "[ERROR] 2024-01-15 16:18:30 payment-service Unable to process payment: no DB connection\n"
            "[ERROR] 2024-01-15 16:07:02 payment-service First DB timeout observed\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 16:18:44 payment-service DB connection timeout: pool exhausted (x89/min)\n"
            "[INFO]  2024-01-15 16:06:50 payment-service Normal operation (2,341 txns/min)\n"
            "[INFO]  2024-01-15 16:06:45 payment-service All systems healthy\n"
        ),
        "warn": "[WARN] 2024-01-15 16:07:30 payment-service DB connection wait time increasing\n",
    },
    "inventory-service": {
        "error": (
            f"[ERROR] 2024-01-15 16:18:45 inventory-service {_DB_TIMEOUT}\n"
            f"[ERROR] 2024-01-15 16:18:44 inventory-service {_DB_TIMEOUT} (x94 last 60s)\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 16:18:45 inventory-service DB connection timeout: pool exhausted (x94/min)\n"
            "[INFO]  2024-01-15 16:07:45 inventory-service Normal operation\n"
        ),
        "warn": "[WARN] 2024-01-15 16:08:00 inventory-service DB pool wait time elevated\n",
    },
    "shipping-service": {
        "error": (
            f"[ERROR] 2024-01-15 16:18:44 shipping-service {_DB_TIMEOUT}\n"
            f"[ERROR] 2024-01-15 16:18:43 shipping-service {_DB_TIMEOUT} (x82 last 60s)\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 16:18:44 shipping-service DB connection timeout: pool exhausted (x82/min)\n"
            "[INFO]  2024-01-15 16:08:45 shipping-service Normal operation\n"
        ),
        "warn": "[WARN] 2024-01-15 16:09:00 shipping-service DB timeouts starting\n",
    },
    "order-service": {
        "error": "No errors in order-service logs. Service is processing requests successfully.\n",
        "warn": (
            "[WARN] 2024-01-15 16:18:44 order-service DBConnAcquisitionRate: 340/min (NORMAL: 45/min) -- 7.5x ABOVE BASELINE\n"
            "[WARN] 2024-01-15 16:18:00 order-service Async batch queries acquiring connections WITHOUT releasing (v4.2.0 regression)\n"
            "[WARN] 2024-01-15 16:15:30 order-service ConnectionAcquisitionRate rising: 280/min\n"
            "[WARN] 2024-01-15 16:10:00 order-service ConnectionAcquisitionRate elevated: 120/min\n"
            "[WARN] 2024-01-15 16:04:00 order-service v4.2.0 startup: async query batching ENABLED\n"
        ),
        "all": (
            "[WARN] 2024-01-15 16:18:44 order-service DBConnAcquisitionRate: 340/min -- 7.5x ABOVE NORMAL (45/min)\n"
            "[WARN] 2024-01-15 16:18:00 order-service Async batch queries acquiring connections without releasing (LEAK)\n"
            "[INFO] 2024-01-15 16:18:40 order-service Processing 890 orders/min (service appears healthy)\n"
            "[INFO] 2024-01-15 16:18:35 order-service No request errors - order-service itself is NOT failing\n"
            "[WARN] 2024-01-15 16:04:00 order-service v4.2.0 startup: async query batching enabled\n"
            "[INFO] 2024-01-15 16:03:55 order-service Deployment v4.2.0 applied\n"
        ),
    },
    "db-pool": {
        "error": (
            "[ERROR] 2024-01-15 16:18:44 db-pool POOL EXHAUSTED: 200/200 connections active, 847 waiting\n"
            "[ERROR] 2024-01-15 16:15:00 db-pool POOL EXHAUSTED: 200/200 connections active, 412 waiting\n"
            "[ERROR] 2024-01-15 16:11:00 db-pool Pool nearing exhaustion: 196/200 connections\n"
            "[WARN]  2024-01-15 16:07:00 db-pool Connection usage >90%: 183/200 active\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 16:18:44 db-pool POOL EXHAUSTED: 200/200 active, 847 waiting\n"
            "[INFO]  2024-01-15 16:04:30 db-pool Connection count rising rapidly since 16:04 -- order-service v4.2.0 deployed at 16:03\n"
            "[INFO]  2024-01-15 16:03:50 db-pool NORMAL STATE: 42/200 connections\n"
        ),
        "warn": "[WARN] 2024-01-15 16:07:00 db-pool Connection usage crossing 90%\n",
    },
    "web-frontend": {
        "all": (
            "[INFO] 2024-01-15 16:18:15 web-frontend 2,890 req/sec - winter-sale campaign active\n"
            "[INFO] 2024-01-15 16:18:10 web-frontend CPU 78% - serving cached content, no DB dependency\n"
        ),
        "error": "No errors in web-frontend. CPU is marketing campaign traffic.\n",
        "warn": "[WARN] 2024-01-15 16:18:00 web-frontend CPU 78% (expected: winter-sale campaign)\n",
    },
}

_RUNBOOKS = {
    "db-pool": (
        "=== RUNBOOK: db-pool ===\n\n"
        "POOL EXHAUSTION — CONNECTION LEAK\n"
        "Detection:\n"
        "  - connections_used near 100% with growing wait_queue\n"
        "  - One service has db_connections_acquired >> normal rate\n"
        "    (that service is the leak source)\n\n"
        "Fix sequence:\n"
        "  a) Identify leaking service: query_metrics(<service>, db_connections_acquired)\n"
        "     Normal rate: ~45/min. Leaking service: significantly higher.\n"
        "  b) Stop the leak: execute_remediation(restart, <leaking-service>)\n"
        "  c) Clear the wait queue: execute_remediation(scale, db-pool)\n"
        "  d) Verify: connections_used should drop within 2 minutes.\n\n"
        "WARNING: Restarting downstream services (payment, inventory, shipping)\n"
        "only provides temporary relief if there is a leak — pool will re-exhaust.\n"
        "Always fix the SOURCE first.\n"
    ),
    "cascade": (
        "=== RUNBOOK: Cascading Failures ===\n"
        "When multiple services fail simultaneously with the SAME error:\n"
        "1. Identify the shared dependency (DB, cache, message queue).\n"
        "2. Check if that dependency is exhausted/degraded.\n"
        "3. Find which upstream service is overloading the dependency.\n"
        "4. Fix the source, THEN fix the dependency.\n"
        "5. Downstream services recover automatically.\n"
        "Key principle: fix root cause before treating symptoms.\n"
    ),
    "order-service": (
        "=== RUNBOOK: order-service ===\n"
        "KNOWN ISSUE v4.2.0: DB Connection Leak\n"
        "  Symptom: db_connections_acquired >> 45/min baseline\n"
        "  Cause: async query batching acquires connections without proper cleanup\n"
        "  Fix: restart or rollback order-service\n"
        "    -> execute_remediation(restart, order-service)\n"
        "  IMPORTANT: also scale db-pool to clear the accumulated wait queue\n"
        "    -> execute_remediation(scale, db-pool)\n"
    ),
    "connection": (
        "=== RUNBOOK: DB Connection Issues ===\n"
        "See runbook: db-pool for pool exhaustion.\n"
        "See runbook: order-service for known v4.2.0 connection leak.\n"
    ),
}

_DEPLOY_HISTORY = {
    "order-service": (
        "=== DEPLOY HISTORY: order-service ===\n"
        "2024-01-15 16:03 | v4.2.0 | ivan  | ACTIVE  | async query batching for order processing performance\n"
        "                                             KNOWN ISSUE: connections not released after batch completion\n"
        "2024-01-14 10:15 | v4.1.9 | julia | RETIRED | minor UI bug fix (stable)\n"
        "\nNOTE: v4.2.0 deployed at 16:03. db-pool exhaustion began at 16:04. Correlation: 1 minute.\n"
    ),
    "payment-service": (
        "=== DEPLOY HISTORY: payment-service ===\n"
        "2024-01-13 09:00 | v3.8.2 | alice | ACTIVE  | dependency update (2 days ago, stable)\n"
        "2024-01-12 14:00 | v3.8.1 | bob   | RETIRED | minor change\n"
    ),
    "inventory-service": (
        "=== DEPLOY HISTORY: inventory-service ===\n"
        "2024-01-12 14:00 | v2.1.0 | carol | ACTIVE  | no recent changes (3 days ago, stable)\n"
    ),
    "shipping-service": (
        "=== DEPLOY HISTORY: shipping-service ===\n"
        "2024-01-11 11:00 | v1.9.3 | dave  | ACTIVE  | no recent changes (4 days ago, stable)\n"
    ),
    "db-pool": (
        "=== DEPLOY HISTORY: db-pool ===\n"
        "No changes in 14 days. db-pool configuration is stable.\n"
    ),
}

CASCADE_CONFIG = IncidentConfig(
    task_id=3,
    severity_level="SEV1",
    root_cause_service="order-service",
    root_cause_type="connection_leak",
    root_cause_keywords=["order-service", "connection", "leak", "db-pool", "v4.2", "async", "exhaust"],
    all_services=["order-service", "db-pool", "payment-service", "inventory-service", "shipping-service", "web-frontend"],
    relevant_services=["db-pool", "order-service", "payment-service"],
    red_herring_services=["web-frontend", "payment-service"],  # payment-service is a symptom, not root cause
    golden_actions=[
        ("restart", "order-service"),  # index 0: stop the leak
        ("scale", "db-pool"),          # index 1: clear the queue
    ],
    action_order_constraints=[(0, 1)],  # restart order-service BEFORE scaling db-pool
    weights={
        "situational": 0.12,
        "diagnostic": 0.28,
        "remediation": 0.22,
        "time": 0.15,
        "communication": 0.10,
        "anti_patterns": 0.13,
    },
    sla_service="payment-service",
    sla_metric="error_rate",
)


class CascadeIncident(BaseIncident):

    def __init__(self, seed: int = 42):
        engine = RewardEngine(CASCADE_CONFIG)
        metrics = make_cascade_metric_engine()
        super().__init__(engine, metrics, CASCADE_CONFIG, seed)
        self._order_fixed = False
        self._pool_scaled = False
        self._downstream_restart_count = 0

    def get_task_id(self) -> int:
        return 3

    def get_initial_context(self) -> str:
        return (
            "=== INCIDENT OPENED: INC-2024-003 | SEV1 - CRITICAL ===\n"
            "Time: 2024-01-15 16:18 UTC\n"
            "Summary: CRITICAL — 3 services are down simultaneously. 8 alerts firing.\n"
            "payment-service, inventory-service, shipping-service all degraded.\n"
            "SLA breached. Every minute of downtime has direct revenue impact.\n\n"
            "WARNING: This is likely a cascading failure. Restarting individual services\n"
            "may provide temporary relief but the issue will recur if root cause is not fixed.\n\n"
            "Find the shared dependency that's failing and the service causing the overload.\n"
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

        is_downstream = any(s in tgt for s in ["payment", "inventory", "shipping"])
        is_order = "order" in tgt
        is_db = "db" in tgt or "pool" in tgt

        if act in ("restart", "rollback") and is_order:
            self._order_fixed = True
            self._after_fix(act, "order-service")
            if self._pool_scaled:
                result = (
                    "SUCCESS: order-service connection leak stopped.\n"
                    "  Connection acquisition rate dropping: 340/min -> 45/min\n"
                    "  Combined with db-pool scaling: all downstream services recovering.\n"
                    "  payment-service: error_rate 89% -> recovering\n"
                    "  All CRITICAL alerts resolving. Incident nearly resolved."
                )
            else:
                result = (
                    "SUCCESS: order-service restarted. Connection leak stopped.\n"
                    "  New connections no longer leaking (rate returning to 45/min)\n"
                    "  BUT: db-pool wait_queue has 847+ connections still queued\n"
                    "  Downstream services still degraded until pool clears.\n"
                    "  Recommend: execute_remediation(scale, db-pool) to clear the wait queue."
                )
        elif act == "scale" and is_db:
            self._pool_scaled = True
            self._after_fix("scale", "db-pool")
            if self._order_fixed:
                result = (
                    "SUCCESS: db-pool scaled to 400 connections.\n"
                    "  Wait queue draining: 847 -> 0 waiting\n"
                    "  All downstream services can acquire connections.\n"
                    "  payment-service recovering, inventory-service recovering, shipping-service recovering.\n"
                    "  Incident resolved. Run verify metrics and declare resolved."
                )
            else:
                result = (
                    "SUCCESS: db-pool scaled to 400 connections.\n"
                    "  Wait queue draining: 847 -> 0\n"
                    "  Downstream services recovering temporarily.\n"
                    "  WARNING: order-service v4.2.0 still leaking at 340/min.\n"
                    "  Pool will re-exhaust in ~5 minutes unless leak is fixed.\n"
                    "  Recommend: execute_remediation(restart, order-service)"
                )
        elif act == "restart" and is_downstream:
            self._downstream_restart_count += 1
            if self._downstream_restart_count <= 2:
                result = (
                    f"SUCCESS: {target} restarted. Temporarily recovering.\n"
                    "  WARNING: db-pool still 100% exhausted (wait queue growing).\n"
                    "  This service will fail again in 1-2 minutes as pool re-exhausts.\n"
                    "  Root cause (connection leak) NOT fixed."
                )
                if self._downstream_restart_count == 2:
                    result += (
                        "\n\n[RECURRENCE ALERT] Services are failing again after restart.\n"
                        "Restarting downstream services is not the solution.\n"
                        "Investigate db-pool and find which service is exhausting it."
                    )
            else:
                result = (
                    f"{target} restarted again, failing immediately.\n"
                    "RECURRENCE: db-pool still exhausted. Fix the connection leak source."
                )
        elif act == "flush_cache":
            result = f"Cache flushed for {target}. No impact on DB connection exhaustion."
        else:
            result = f"Action '{act}' on '{target}': no significant change to DB exhaustion."

        self._record("execute_remediation", args, result, step_reward)
        return result, step_reward

    def declare_resolved(self, root_cause: str) -> tuple[float, str]:
        args = {"root_cause": root_cause}
        step_reward = self._tick("declare_resolved", args, self.step_count)

        # Downstream-only restarts count as wrong actions in the engine
        if self._downstream_restart_count >= 2 and not self._order_fixed:
            self.reward_engine.wrong_actions_count += self._downstream_restart_count

        final_score, breakdown = self.reward_engine.compute_final_reward(
            root_cause, self.step_count
        )
        feedback = breakdown.to_feedback()
        self.done = True
        self._record("declare_resolved", args, feedback, step_reward)
        return final_score, feedback
