"""
Task 5: Alert Storm — Consumer Deadlock (Hard)

Scenario: 11 firing alerts from 6 services. The root cause is a deadlock in
notification-service v2.8.0 (deployed 3 hours ago). It consumed from message-queue
using a JPA transaction that holds a DB lock while awaiting a network callback —
deadlock on every message. Result: 0 messages consumed, message-queue fills up,
all 4 producer services back up and time out.

The critical trap: notification-service HTTP health checks return 200 OK.
From metrics alone it looks healthy. Only logs reveal the DEADLOCK stacktrace.
The agent must NOT fix producers first — they're symptoms.

Unrelated red herrings:
  1. analytics-service — disk alert from failed log rotation 72 hours ago (predates incident)
  2. cdn-proxy — health check flap from brief network hiccup, resolved itself

Ordering constraint: restart notification-service BEFORE restart message-queue.
Restarting message-queue first clears the queue but it refills immediately (consumer
still deadlocked), wasting the fix and extending downtime.

Optimal path (9–12 steps, max_steps=20):
  list_alerts
  -> query_logs(message-queue, error)          [consumer lag: 0 msgs/min consumed]
  -> query_logs(notification-service, error)   [DEADLOCK stacktrace — smoking gun]
  -> check_deploy_history(notification-service) [v2.8.0 3h ago, known deadlock bug]
  -> query_metrics(message-queue)              [97% memory, 847K depth]
  -> update_status(...)
  -> execute_remediation(restart, notification-service)   [fix consumer first]
  -> execute_remediation(restart, message-queue)          [now safe to clear queue]
  -> query_metrics(message-queue)              [verify draining]
  -> declare_resolved(...)

IncidentConfig golden actions: [("restart","notification-service"), ("restart","message-queue")]
Action order constraint: notification-service (0) before message-queue (1)
"""

from .incident_base import BaseIncident
from .reward_engine import RewardEngine, IncidentConfig
from .metric_engine import make_alert_storm_metric_engine
from .belief_engine import (
    BeliefEngine,
    ALERT_STORM_HYPOTHESIS_SPACE,
    ALERT_STORM_TRUE_HYPOTHESIS,
    ALERT_STORM_LIKELIHOOD_TABLE,
)
from .workflow_machine import WorkflowMachine

_INITIAL_ALERTS = """
=== FIRING ALERTS (11 active) ===
[CRITICAL] message-queue        | Memory          | 97.0%    | threshold: 85%   | duration: 31m
[CRITICAL] message-queue        | QueueDepth      | 847K msgs| threshold: 100K  | duration: 28m
[HIGH]     order-service         | ErrorRate       | 34.2%    | threshold: 5%    | duration: 29m
[HIGH]     order-service         | Latency_P99     | 30000ms  | threshold: 500ms | duration: 29m
[HIGH]     payment-service       | ErrorRate       | 31.8%    | threshold: 5%    | duration: 29m
[HIGH]     payment-service       | Latency_P99     | 30000ms  | threshold: 500ms | duration: 29m
[HIGH]     email-service         | ErrorRate       | 28.4%    | threshold: 5%    | duration: 28m
[MEDIUM]   user-service          | ErrorRate       | 22.1%    | threshold: 10%   | duration: 27m
[MEDIUM]   notification-service  | CPUUsage        | 45%      | threshold: 40%   | duration: 27m
[LOW]      analytics-service     | DiskUsage       | 89%      | threshold: 80%   | duration: 72h  | unresolved
[LOW]      cdn-proxy             | HealthCheck     | FLAPPING | threshold: 99%   | duration: 4m   | flapping
"""

_LOGS = {
    "message-queue": {
        "error": (
            "[ERROR] 2024-01-15 14:52:18 message-queue Consumer idle: notification-service — 0 messages/min (31 min)\n"
            "    consumer_group: notification-consumers | expected_rate: 1200/min | actual_rate: 0/min\n"
            "    consumer_count: 4 registered, 4 IDLE (not processing)\n"
            "[ERROR] 2024-01-15 14:51:44 message-queue Queue depth exceeded 800K: notifications.delivery (847,221 messages)\n"
            "[ERROR] 2024-01-15 14:50:12 message-queue Memory pressure: 97.0% heap — approaching OOM threshold (99%)\n"
            "[WARN]  2024-01-15 14:49:38 message-queue 4 producers (order, payment, email, user) backing up — publish latency 30s\n"
            "[ERROR] 2024-01-15 14:21:33 message-queue Consumer notification-service: 0 acks received since 14:21 (consumer went idle)\n"
            "[INFO]  2024-01-15 14:21:00 message-queue notification-service consumers registered: 4 threads\n"
            "[INFO]  2024-01-15 13:58:12 message-queue Pre-incident: processing 1,240 notifications/min — healthy\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 14:52:00 message-queue Consumer group notification-consumers: lag 847,221 messages\n"
            "[WARN] 2024-01-15 14:50:30 message-queue Memory 97%: dead-letter queue filling\n"
            "[WARN] 2024-01-15 14:49:38 message-queue Producer backpressure: 4 services timing out on publish\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:52:18 message-queue Consumer idle: notification-service — 0 msgs/min consumed (31 min)\n"
            "[ERROR] 2024-01-15 14:51:44 message-queue Queue depth 847K — memory 97%\n"
            "[ERROR] 2024-01-15 14:50:12 message-queue Memory pressure 97.0% heap\n"
            "[WARN]  2024-01-15 14:49:38 message-queue Producer backpressure: order, payment, email, user timing out\n"
            "[INFO]  2024-01-15 14:21:00 message-queue notification-service consumers registered OK (4 threads)\n"
            "[INFO]  2024-01-15 13:58:00 message-queue Healthy — 1,240 notifications/min processed\n"
        ),
    },
    "notification-service": {
        "error": (
            "[ERROR] 2024-01-15 14:21:44 notification-service DEADLOCK detected in NotificationProcessor\n"
            "    Thread[notification-consumer-1] holding DB lock on notifications.pending, waiting for network callback\n"
            "    Thread[notification-consumer-2] holding network slot, waiting for DB lock on notifications.pending\n"
            "    java.lang.Thread.State: BLOCKED (on object monitor)\n"
            "    at com.notification.processor.NotificationProcessor.processWithAck(NotificationProcessor.java:184)\n"
            "    at com.notification.jpa.TransactionManager.commitWithCallback(TransactionManager.java:92)\n"
            "    DEADLOCK affects all 4 consumer threads — no messages will be processed\n"
            "    This is a known issue in v2.8.0: JPA transaction holds DB lock across network calls.\n"
            "    Introduced in v2.8.0 changelog: 'Added transactional ack to prevent duplicate delivery'\n"
            "[ERROR] 2024-01-15 14:21:42 notification-service All 4 consumer threads BLOCKED — deadlock\n"
            "[ERROR] 2024-01-15 14:21:40 notification-service Deadlock on first message after startup (v2.8.0)\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 14:52:10 notification-service CPU 45%: threads blocked, CPU wasted on mutex contention\n"
            "[WARN] 2024-01-15 14:22:00 notification-service Consumer threads not processing — check for deadlock\n"
            "[WARN] 2024-01-15 14:21:44 notification-service DB lock wait timeout exceeded (30s) on consumer thread-1\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:21:44 notification-service DEADLOCK: all consumer threads BLOCKED (v2.8.0 bug)\n"
            "[ERROR] 2024-01-15 14:21:42 notification-service DB lock held across network callback — known v2.8.0 issue\n"
            "[WARN]  2024-01-15 14:52:10 notification-service CPU 45% (mutex contention), HTTP health: 200 OK\n"
            "[INFO]  2024-01-15 14:21:00 notification-service v2.8.0 deployed — consumers started, then deadlocked\n"
            "[INFO]  2024-01-15 14:21:00 notification-service Consumer threads registered with message-queue\n"
            "[INFO]  2024-01-15 13:58:00 notification-service v2.7.9 healthy — processing 1,240/min\n"
        ),
    },
    "order-service": {
        "error": (
            "[ERROR] 2024-01-15 14:52:15 order-service PublishTimeout: message-queue unavailable (timeout: 30s)\n"
            "    at com.order.events.OrderEventPublisher.publish(OrderEventPublisher.java:77)\n"
            "    Caused by: com.rabbitmq.client.AlreadyClosedException: connection closed — broker OOM pressure\n"
            "[ERROR] 2024-01-15 14:52:10 order-service Order confirmation notifications failing (x148 last 2min)\n"
            "[WARN]  2024-01-15 14:52:08 order-service message-queue backpressure: 30s publish timeout\n"
            "NOTE: order-service itself is healthy — this is a MESSAGE-QUEUE upstream failure.\n"
            "      Restarting order-service will NOT fix the queue issue.\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:52:15 order-service PublishTimeout: message-queue (x148 last 2min)\n"
            "[WARN]  2024-01-15 14:52:08 order-service Backpressure from message-queue\n"
            "[INFO]  2024-01-15 14:52:00 order-service Core order processing: 148 req/min (healthy)\n"
            "[INFO]  2024-01-15 13:58:00 order-service Healthy — all upstream dependencies OK\n"
        ),
        "warn": "[WARN] 2024-01-15 14:52:08 order-service message-queue backpressure: 30s publish timeout\n",
    },
    "payment-service": {
        "error": (
            "[ERROR] 2024-01-15 14:52:12 payment-service PublishTimeout: message-queue (payment receipts)\n"
            "    Caused by: com.rabbitmq.client.AlreadyClosedException: broker memory alarm\n"
            "[ERROR] 2024-01-15 14:52:08 payment-service Payment receipts backed up: 847 pending (queue full)\n"
            "NOTE: payment-service is healthy — symptom of message-queue being full.\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:52:12 payment-service PublishTimeout: message-queue (x92 last 2min)\n"
            "[WARN]  2024-01-15 14:52:05 payment-service Backpressure: message-queue memory alarm\n"
            "[INFO]  2024-01-15 14:52:00 payment-service Payment processing: 38 txn/min (healthy)\n"
            "[INFO]  2024-01-15 13:58:00 payment-service Healthy\n"
        ),
        "warn": "[WARN] 2024-01-15 14:52:05 payment-service Backpressure from message-queue memory alarm\n",
    },
    "email-service": {
        "error": (
            "[ERROR] 2024-01-15 14:52:14 email-service PublishTimeout: message-queue (email queuing)\n"
            "[ERROR] 2024-01-15 14:52:09 email-service Email delivery pipeline stalled: publisher backing up\n"
            "NOTE: email-service is healthy — symptom of full message-queue.\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:52:14 email-service PublishTimeout: message-queue (x240 last 2min)\n"
            "[WARN]  2024-01-15 14:52:07 email-service Email queue stalled — upstream message-queue full\n"
            "[INFO]  2024-01-15 13:58:00 email-service Healthy — 240 emails/min processed\n"
        ),
        "warn": "[WARN] 2024-01-15 14:52:07 email-service email queue stalled — upstream message-queue full\n",
    },
    "user-service": {
        "error": (
            "[ERROR] 2024-01-15 14:52:11 user-service PublishTimeout: message-queue (user-event notifications)\n"
            "NOTE: user-service is healthy — symptom of full message-queue.\n"
        ),
        "all": (
            "[ERROR] 2024-01-15 14:52:11 user-service PublishTimeout: message-queue (x412 last 2min)\n"
            "[INFO]  2024-01-15 14:52:00 user-service Core user operations: healthy (412 req/min)\n"
            "[INFO]  2024-01-15 13:58:00 user-service Healthy\n"
        ),
        "warn": "[WARN] 2024-01-15 14:51:58 user-service message-queue not accepting publishes\n",
    },
    "analytics-service": {
        "error": "No errors in analytics-service. Disk alert is from a stale log file not rotated.\n",
        "all": (
            "[WARN]  2024-01-15 14:52:00 analytics-service Disk usage 89%: /var/log/analytics (log rotation failed 72h ago)\n"
            "[INFO]  2024-01-15 14:52:00 analytics-service Processing 8,400 events/min — healthy\n"
            "[INFO]  2024-01-15 14:52:00 analytics-service UNRELATED TO CURRENT INCIDENT: disk alert predates incident by 71.5 hours\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 14:52:00 analytics-service Disk 89%: stale log file (log rotation cron failed 72h ago)\n"
            "NOTE: This alert is UNRELATED to the message-queue incident.\n"
        ),
    },
    "cdn-proxy": {
        "error": "No errors in cdn-proxy. Health check flap was a 4-minute network hiccup — self-resolved.\n",
        "all": (
            "[INFO] 2024-01-15 14:52:00 cdn-proxy Health check: PASSING (flap resolved 2min ago)\n"
            "[WARN] 2024-01-15 14:48:00 cdn-proxy Health check FAILED (network hiccup — transient)\n"
            "[INFO] 2024-01-15 14:52:00 cdn-proxy UNRELATED TO CURRENT INCIDENT: network blip, self-resolved\n"
        ),
        "warn": (
            "[WARN] 2024-01-15 14:48:00 cdn-proxy Health check FAILED (transient network hiccup, 4m ago)\n"
            "NOTE: cdn-proxy flap is UNRELATED to message-queue or producer failures.\n"
        ),
    },
}

_RUNBOOKS = {
    "message-queue": (
        "=== RUNBOOK: message-queue (RabbitMQ) ===\n\n"
        "SYMPTOM: Queue depth growing, memory alarm, producers timing out.\n\n"
        "TRIAGE ORDER:\n"
        "  1. Identify which CONSUMER group is lagging (query_logs(message-queue)).\n"
        "  2. Check the consumer service logs for errors (deadlock, OOM, crash).\n"
        "  3. Fix the consumer FIRST before clearing the queue.\n"
        "     REASON: If you clear the queue while consumer is still broken,\n"
        "             the queue will refill within minutes from producer backlog.\n"
        "  4. After consumer is fixed: execute_remediation(restart, message-queue)\n"
        "     This clears accumulated memory and resets producer connections.\n\n"
        "QUEUE DEPTH ALARM (>100K messages):\n"
        "  Root cause: consumer stopped processing.\n"
        "  Fix: fix consumer, THEN restart message-queue.\n"
        "  Do NOT restart message-queue first — see above.\n\n"
        "MEMORY ALARM (>85%):\n"
        "  Caused by queue depth. Fix queue depth to resolve memory alarm.\n\n"
        "ESCALATION: Page #platform-messaging if memory > 99% or restart fails.\n"
    ),
    "notification-service": (
        "=== RUNBOOK: notification-service ===\n\n"
        "v2.8.0 KNOWN ISSUE (2024-01-15):\n"
        "  DEADLOCK in NotificationProcessor.processWithAck() introduced by\n"
        "  transactional ack feature: JPA transaction holds DB lock across async\n"
        "  network callbacks, causing thread deadlock on first message processed.\n\n"
        "  Symptoms:\n"
        "    - Consumer threads BLOCKED (0 messages/min consumed)\n"
        "    - HTTP health check still returns 200 OK (misleading)\n"
        "    - CPU elevated from mutex contention\n"
        "    - DEADLOCK stacktrace in error logs\n\n"
        "  Fix: execute_remediation(restart, notification-service)\n"
        "    - Restart clears the deadlocked threads\n"
        "    - Service comes back on v2.8.0 (hotfix required for permanent fix)\n"
        "    - Processing resumes at 1,200/min after restart\n\n"
        "  IMPORTANT: Fix notification-service BEFORE restarting message-queue.\n"
        "             If you restart message-queue first, the queue refills immediately.\n\n"
        "v2.7.9 (stable): No deadlock issue. Hotfix v2.8.1 pending.\n\n"
        "ESCALATION: Page #notifications-oncall if restart fails.\n"
    ),
    "alert-storm": (
        "=== RUNBOOK: Alert Storm Triage ===\n\n"
        "WHEN MANY ALERTS FIRE SIMULTANEOUSLY:\n"
        "  1. Group alerts by service type: consumers, producers, infrastructure\n"
        "  2. Look for the SINGLE ROOT CAUSE that explains all alerts\n"
        "     - If 4 producers all fail at the same time: suspect shared dependency\n"
        "     - Check the queue/broker they all publish to\n"
        "  3. Ignore alerts with long duration (72h+) or unrelated patterns\n\n"
        "QUEUE + CONSUMER DEADLOCK PATTERN:\n"
        "  - Queue memory/depth alarms\n"
        "  - Multiple producer timeouts\n"
        "  - Consumer service looks 'healthy' from metrics but logs show deadlock\n"
        "  Fix order: restart consumer -> verify queue draining -> restart queue broker\n\n"
        "ANTI-PATTERNS:\n"
        "  - Do NOT restart all failing services (symptoms will recur)\n"
        "  - Do NOT clear the queue before fixing the consumer\n"
        "  - Do NOT chase unrelated alerts during active incident\n"
    ),
    "order-service": (
        "=== RUNBOOK: order-service ===\n\n"
        "PublishTimeout to message-queue:\n"
        "  This means the message broker is full or unavailable.\n"
        "  Fix: investigate and fix message-queue, NOT order-service.\n"
        "  Restarting order-service will not resolve the message-queue issue.\n"
    ),
    "payment-service": (
        "=== RUNBOOK: payment-service ===\n\n"
        "PublishTimeout to message-queue:\n"
        "  Fix: investigate message-queue — it is the upstream dependency.\n"
        "  Payment processing itself is healthy.\n"
    ),
    "analytics-service": (
        "=== RUNBOOK: analytics-service ===\n\n"
        "DISK USAGE ALERT:\n"
        "  Caused by failed log rotation cron job. This is UNRELATED to message-queue incident.\n"
        "  Fix: ssh analytics-node && logrotate -f /etc/logrotate.d/analytics\n"
        "  Do NOT treat this as part of the current incident.\n"
    ),
}

_DEPLOY_HISTORY = {
    "notification-service": (
        "=== DEPLOY HISTORY: notification-service ===\n"
        "2024-01-15 11:21 | v2.8.0 | carol | ACTIVE  | Added transactional ack for duplicate prevention\n"
        "                   CHANGELOG: JPA transaction now wraps full ack+network callback cycle\n"
        "                   REGRESSION: holds DB lock across async network call → deadlock (unforeseen)\n"
        "2024-01-14 16:30 | v2.7.9 | carol | RETIRED | Performance improvements (stable)\n"
        "2024-01-12 09:00 | v2.7.8 | carol | RETIRED | Retry logic improvements\n"
        "\nNOTE: v2.8.0 deployed at 11:21 UTC (3h before incident onset at 14:21 UTC).\n"
        "      Deadlock occurs on first message consumed after startup.\n"
        "      v2.7.9 was stable for 1 day. Recommend rollback or restart (immediate relief).\n"
    ),
    "message-queue": (
        "=== DEPLOY HISTORY: message-queue ===\n"
        "2023-12-16 10:00 | v3.11.2 | ops-team | ACTIVE | security patch (30 days ago, stable)\n"
        "\nNOTE: No recent message-queue changes. Queue filling is caused by consumer deadlock.\n"
    ),
    "order-service": (
        "=== DEPLOY HISTORY: order-service ===\n"
        "2024-01-11 14:00 | v7.4.1 | alice | ACTIVE | Order state machine refactor (4 days ago, stable)\n"
    ),
    "payment-service": (
        "=== DEPLOY HISTORY: payment-service ===\n"
        "2024-01-10 11:00 | v6.2.0 | bob | ACTIVE | Fraud detection update (5 days ago, stable)\n"
    ),
    "email-service": (
        "=== DEPLOY HISTORY: email-service ===\n"
        "2024-01-09 09:00 | v4.1.3 | dave | ACTIVE | Template engine update (6 days ago, stable)\n"
    ),
    "user-service": (
        "=== DEPLOY HISTORY: user-service ===\n"
        "2024-01-13 15:00 | v5.3.2 | eve | ACTIVE | Profile cache improvements (2 days ago, stable)\n"
    ),
    "analytics-service": (
        "=== DEPLOY HISTORY: analytics-service ===\n"
        "2024-01-08 10:00 | v2.1.0 | ops | ACTIVE | Event schema migration (7 days ago, stable)\n"
        "NOTE: Disk alert predates incident by 72 hours — unrelated.\n"
    ),
    "cdn-proxy": (
        "=== DEPLOY HISTORY: cdn-proxy ===\n"
        "2023-12-01 08:00 | v1.9.4 | ops | ACTIVE | TLS cipher update (45 days ago, stable)\n"
    ),
}

ALERT_STORM_CONFIG = IncidentConfig(
    task_id=5,
    severity_level="SEV1",
    root_cause_service="notification-service",
    root_cause_type="consumer_deadlock",
    root_cause_keywords=["deadlock", "notification-service", "message-queue", "consumer", "v2.8.0", "jpa", "transaction"],
    all_services=[
        "message-queue", "notification-service",
        "order-service", "payment-service", "email-service", "user-service",
        "analytics-service", "cdn-proxy",
    ],
    relevant_services=["notification-service", "message-queue", "order-service", "payment-service"],
    red_herring_services=["analytics-service", "cdn-proxy"],
    golden_actions=[("restart", "notification-service"), ("restart", "message-queue")],
    action_order_constraints=[(0, 1)],   # notification-service restart must come before message-queue restart
    weights={
        "situational":        0.08,
        "diagnostic":         0.17,
        "remediation":        0.26,
        "time":               0.09,
        "communication":      0.10,
        "anti_patterns":      0.13,    # higher: 11 alerts tempt wrong fixes
        "epistemic_quality":  0.10,    # higher: more hypotheses require more belief work
        "workflow_coherence": 0.07,
    },
    hypothesis_space=ALERT_STORM_HYPOTHESIS_SPACE,
    likelihood_table=ALERT_STORM_LIKELIHOOD_TABLE,
    true_hypothesis=ALERT_STORM_TRUE_HYPOTHESIS,
    sla_service="payment-service",
    sla_metric="error_rate",
    max_steps=20,
)


class AlertStormIncident(BaseIncident):

    def __init__(self, seed: int = 42):
        engine = RewardEngine(ALERT_STORM_CONFIG)
        metrics = make_alert_storm_metric_engine()
        belief = BeliefEngine(
            ALERT_STORM_HYPOTHESIS_SPACE,
            ALERT_STORM_LIKELIHOOD_TABLE,
            ALERT_STORM_TRUE_HYPOTHESIS,
        )
        workflow = WorkflowMachine(root_cause_service="notification-service")
        super().__init__(engine, metrics, ALERT_STORM_CONFIG, belief, workflow, seed)
        self._notif_restarted = False
        self._queue_restarted = False
        self._wrong_restarts: list[str] = []   # producer restarts that did nothing

    def get_task_id(self) -> int:
        return 5

    def get_initial_context(self) -> str:
        return (
            "=== INCIDENT OPENED: INC-2024-005 | SEV1 ===\n"
            "Time: 2024-01-15 14:52 UTC\n"
            "Summary: 11 alerts firing. Multiple services reporting high error rates\n"
            "and timeouts. message-queue memory at 97%. Customers are affected.\n\n"
            "Warning: This is a noisy incident with many simultaneous alerts.\n"
            "Not all alerts represent root causes. Identify the single root cause\n"
            "before attempting fixes.\n\n"
            "Tip: When many services fail at once, they may share a common dependency.\n"
            "Look for a service that all failing services depend on.\n"
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

        # Enrich message-queue metrics with consumer processing rate
        if svc == "message-queue" and met == "all":
            base = self.metric_engine.format_all(svc)
            notif_processing = self.metric_engine.get_raw("notification-service", "db_connections_acquired") or 0.0
            result = (
                base + "\n"
                f"  consumer_processing_rate: {notif_processing:.0f} msgs/min (notification-service)\n"
                f"  NOTE: healthy baseline = 1200 msgs/min; current = {notif_processing:.0f} msgs/min"
            )
        elif svc == "notification-service" and met == "all":
            base = self.metric_engine.format_all(svc)
            processing_rate = self.metric_engine.get_raw("notification-service", "db_connections_acquired") or 0.0
            result = (
                base + "\n"
                f"  processing_rate: {processing_rate:.0f} msgs/min  ← 0 indicates consumer deadlock\n"
                "  http_health: 200 OK  ← NOTE: health check passes despite deadlock\n"
                "  consumer_threads: 4 registered, 4 BLOCKED (from logs)\n"
            )
        elif met == "all":
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

        if act == "restart" and "notification" in tgt:
            self._notif_restarted = True
            self._after_fix("restart", "notification-service")
            result = (
                "SUCCESS: notification-service restarted.\n"
                "  Deadlocked threads cleared. New consumer threads starting.\n"
                "  Consumer reconnecting to message-queue...\n"
                "  Processing rate recovering: 0 -> 450 msgs/min (and growing)\n"
                "  message-queue: consumer lag beginning to decrease\n"
                "  Queue depth: 847K (still high, but now draining at 450 msgs/min)\n"
                "  Estimated time to drain at current rate: ~31 minutes\n"
                "NEXT STEP: restart message-queue to accelerate queue drain and\n"
                "  release producer connections that are still backed up.\n"
                "Recommend: execute_remediation(restart, message-queue)"
            )
        elif act == "restart" and "message-queue" in tgt:
            if self._notif_restarted:
                # Correct order: consumer fixed first, now safe to clear queue
                self._queue_restarted = True
                self._after_fix("restart", "message-queue")
                result = (
                    "SUCCESS: message-queue restarted.\n"
                    "  Queue cleared: 847K messages flushed (per retention policy: dead-letter)\n"
                    "  Memory: 97% -> 22% (immediate relief)\n"
                    "  Producer connections reset: order, payment, email, user-service reconnecting\n"
                    "  Producer error rates dropping rapidly.\n"
                    "  notification-service consumer reconnecting to fresh queue.\n"
                    "  error_rate recovering: order(34% -> 8%), payment(31% -> 6%), email(28% -> 4%), user(22% -> 2%)\n"
                    "Recommend: verify metrics then declare resolved."
                )
            else:
                # Wrong order: queue cleared but consumer still deadlocked
                self._queue_restarted = True
                self._after_fix("restart", "message-queue")
                result = (
                    "message-queue restarted.\n"
                    "  Queue temporarily cleared: 847K -> 0 messages.\n"
                    "  Memory drops: 97% -> 18%.\n"
                    "  PROBLEM: notification-service consumer is STILL DEADLOCKED.\n"
                    "  [30s later] Queue depth growing: 0 -> 12K messages (producers refilling)\n"
                    "  [60s later] Queue depth: 24K. Memory climbing: 18% -> 20%\n"
                    "  NOTICE: Queue will re-exhaust in ~25 minutes — consumer not consuming.\n"
                    "  producer error rates STILL HIGH: order(34%), payment(31%), email(28%)\n"
                    "  CRITICAL: You MUST fix the consumer (notification-service) first.\n"
                    "  execute_remediation(restart, notification-service) IMMEDIATELY"
                )
        elif act == "restart" and any(p in tgt for p in ["order", "payment", "email", "user"]):
            self._wrong_restarts.append(tgt)
            self._after_fix(act, tgt)
            result = (
                f"{tgt} restarted.\n"
                f"  {tgt} back online in 10 seconds.\n"
                f"  BRIEF IMPROVEMENT: error_rate drops ~15% for 30-60 seconds.\n"
                f"  REGRESSION: error_rate climbing back — message-queue still full.\n"
                f"  NOTICE: {tgt} errors are caused by message-queue backpressure, not {tgt} itself.\n"
                f"  Restarting producers does not fix the queue. Fix notification-service first.\n"
                f"  Root cause still active."
            )
        elif act == "scale" and "message-queue" in tgt:
            result = (
                "message-queue scaled to additional nodes.\n"
                "  Memory headroom increased: 97% -> 74% (temporary relief).\n"
                "  Queue depth still growing: consumer (notification-service) still deadlocked.\n"
                "  Scaling does not fix a consumer deadlock. Fix notification-service.\n"
                "  Root cause still active."
            )
        else:
            result = f"Action '{act}' on '{target}': completed. No significant change to active alerts."

        self._record("execute_remediation", args, result, step_reward)
        return result, step_reward

    def declare_resolved(self, root_cause: str) -> tuple[float, str]:
        args = {"root_cause": root_cause}
        step_reward = self._tick("declare_resolved", args, self.step_count)

        # Extra penalties for wrong remediations
        for svc in self._wrong_restarts:
            self.reward_engine.wrong_actions_count += 1

        final_score, breakdown = self.reward_engine.compute_final_reward(
            root_cause, self.step_count, self.belief_engine, self.workflow_machine
        )
        feedback = breakdown.to_feedback()
        self.done = True
        self._record("declare_resolved", args, feedback, step_reward)
        return final_score, feedback
