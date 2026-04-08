"""
Metric Evolution Engine — makes the incident environment "alive."

Each step, metrics drift based on the active incident state.
Agent remediations modify the drift. Querying metrics returns current evolved values.
This creates genuine time pressure: if the agent does nothing, things get worse.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ServiceMetrics:
    """Current metric snapshot for one service."""
    error_rate: float = 0.0        # 0–100 %
    latency_p99: float = 0.0       # ms
    memory_usage: float = 0.0      # 0–100 %
    cpu_usage: float = 0.0         # 0–100 %
    requests_per_sec: float = 0.0
    connections_used: float = 0.0  # for db-pool: connections 0–200+
    wait_queue: float = 0.0        # db-pool: requests waiting
    db_connections_acquired: float = 0.0  # rate per min


class MetricEngine:
    """
    Holds and evolves service metrics over episode steps.

    Usage:
        engine = MetricEngine(initial, evolution_fn, sla_check_fn)
        engine.apply_fix("restart", "payment-service")  # before first step
        for step in episode:
            engine.tick()                          # evolve metrics
            display = engine.get_display(service)  # what agent sees
            sla_ok = engine.is_sla_ok()
    """

    def __init__(
        self,
        initial: dict[str, ServiceMetrics],
        evolution_fn: Callable[[dict[str, ServiceMetrics], int, set[str]], None],
        sla_check_fn: Callable[[dict[str, ServiceMetrics]], bool],
    ):
        self._metrics = initial
        self._evolution_fn = evolution_fn
        self._sla_check_fn = sla_check_fn
        self._step = 0
        self._fixes_applied: set[str] = set()  # "action:target" tokens

    def tick(self) -> None:
        """Advance one episode step — metrics evolve."""
        self._step += 1
        self._evolution_fn(self._metrics, self._step, self._fixes_applied)

    def apply_fix(self, action: str, target: str) -> None:
        """Record that a remediation was applied; evolution_fn can query this."""
        self._fixes_applied.add(f"{action}:{target}")

    def is_sla_ok(self) -> bool:
        """Returns True if SLA is currently satisfied."""
        return self._sla_check_fn(self._metrics)

    def get_display(self, service: str) -> dict[str, str]:
        """
        Return human-readable metric strings for a service (as the agent sees them).
        Values are the current evolved values.
        """
        m = self._metrics.get(service)
        if m is None:
            return {}
        out = {}
        if m.error_rate > 0:
            out["error_rate"] = f"{m.error_rate:.1f}%"
        if m.latency_p99 > 0:
            out["latency_p99"] = f"{m.latency_p99:.0f}ms"
        if m.memory_usage > 0:
            out["memory_usage"] = f"{m.memory_usage:.1f}%"
        if m.cpu_usage > 0:
            out["cpu_usage"] = f"{m.cpu_usage:.1f}%"
        if m.requests_per_sec > 0:
            out["requests_per_sec"] = f"{m.requests_per_sec:.0f}/sec"
        if m.connections_used > 0:
            out["connections_used"] = f"{int(m.connections_used)}/200"
        if m.wait_queue > 0:
            out["wait_queue"] = f"{int(m.wait_queue)} waiting"
        if m.db_connections_acquired > 0:
            out["db_connections_acquired"] = f"{m.db_connections_acquired:.0f}/min"
        return out

    def get_raw(self, service: str, metric: str) -> float | None:
        """Return raw float for a specific metric (for SLA checks)."""
        m = self._metrics.get(service)
        if m is None:
            return None
        return getattr(m, metric, None)

    def format_metric(self, service: str, metric: str) -> str:
        """Return a formatted metric string for a specific metric."""
        display = self.get_display(service)
        return display.get(metric, f"metric '{metric}' not available for {service}")

    def format_all(self, service: str) -> str:
        """Return all metrics for a service as a formatted string."""
        display = self.get_display(service)
        if not display:
            return f"No metrics available for '{service}'."
        lines = [f"=== LIVE METRICS: {service} (step {self._step}) ==="]
        for k, v in display.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1 — OOM incident metric engine factory
# ---------------------------------------------------------------------------

def make_oom_metric_engine() -> MetricEngine:
    """
    payment-service memory leaks at ~1.8%/step.
    At 100%: service crashes, error_rate spikes.
    Restart resets memory to 18%.
    api-gateway shows elevated CPU (68%) AND elevated latency (1850ms) to be a more
    convincing red herring — agents must query logs to rule it out, not just metrics.
    """
    initial = {
        "payment-service": ServiceMetrics(
            error_rate=3.2, latency_p99=4200, memory_usage=97.0,
            cpu_usage=62.0, requests_per_sec=38.0,
        ),
        "api-gateway": ServiceMetrics(
            # Strengthened red herring: elevated CPU matches alert, plus latency elevation
            # from slow upstream payment-service responses propagating back
            error_rate=0.02, latency_p99=1850, memory_usage=41.0,
            cpu_usage=68.0, requests_per_sec=420.0,
        ),
        "user-service": ServiceMetrics(
            error_rate=0.01, latency_p99=78, memory_usage=38.0,
            cpu_usage=22.0, requests_per_sec=210.0,
        ),
    }

    def evolve(metrics: dict[str, ServiceMetrics], step: int, fixes: set[str]) -> None:
        ps = metrics["payment-service"]
        if "restart:payment-service" in fixes:
            # Recovering: memory dropped, metrics normalizing
            ps.memory_usage = max(18.0, ps.memory_usage - 8.0)
            ps.error_rate = max(0.1, ps.error_rate - 1.2)
            ps.latency_p99 = max(95, ps.latency_p99 - 600)
            ps.requests_per_sec = min(220, ps.requests_per_sec + 30)
        else:
            # Leaking: memory grows toward crash
            ps.memory_usage = min(100.0, ps.memory_usage + 1.8)
            ps.error_rate = min(35.0, ps.error_rate + 1.1)
            ps.latency_p99 = min(30000, ps.latency_p99 + 800)
            ps.requests_per_sec = max(0, ps.requests_per_sec - 6)
            if ps.memory_usage >= 100.0:
                # Crash: downstream services start seeing errors
                metrics["api-gateway"].error_rate = min(15.0, metrics["api-gateway"].error_rate + 2.0)

    def sla_ok(metrics: dict[str, ServiceMetrics]) -> bool:
        return metrics["payment-service"].error_rate < 5.0

    return MetricEngine(initial, evolve, sla_ok)


# ---------------------------------------------------------------------------
# Task 2 — Bad deploy metric engine factory
# ---------------------------------------------------------------------------

def make_deploy_metric_engine() -> MetricEngine:
    """
    api-gateway error_rate stuck high due to code bug (stable, not growing).
    But circuit breakers worsen over time if not fixed.
    Rollback restores normal operation.
    """
    initial = {
        "api-gateway": ServiceMetrics(
            error_rate=12.4, latency_p99=2350, memory_usage=44.0,
            cpu_usage=38.0, requests_per_sec=31.0,
        ),
        "user-service": ServiceMetrics(
            error_rate=0.03, latency_p99=78, memory_usage=36.0,
            cpu_usage=21.0, requests_per_sec=201.0,
        ),
        "payment-service": ServiceMetrics(
            error_rate=0.1, latency_p99=112, memory_usage=42.0,
            cpu_usage=25.0, requests_per_sec=18.0,
        ),
        "web-frontend": ServiceMetrics(
            # Red herring: slightly elevated CPU but unrelated to the incident
            error_rate=0.05, latency_p99=180, memory_usage=55.0,
            cpu_usage=71.0, requests_per_sec=890.0,
        ),
    }

    def evolve(metrics: dict[str, ServiceMetrics], step: int, fixes: set[str]) -> None:
        gw = metrics["api-gateway"]
        if "rollback:api-gateway" in fixes:
            gw.error_rate = max(0.1, gw.error_rate - 3.0)
            gw.latency_p99 = max(90, gw.latency_p99 - 450)
            gw.requests_per_sec = min(43, gw.requests_per_sec + 3)
            metrics["payment-service"].requests_per_sec = min(38, metrics["payment-service"].requests_per_sec + 4)
        else:
            # Circuit breakers degrade further as errors accumulate
            if step > 4:
                metrics["payment-service"].error_rate = min(8.0, metrics["payment-service"].error_rate + 0.3)
                metrics["payment-service"].requests_per_sec = max(5, metrics["payment-service"].requests_per_sec - 1)
            # Misleading: web-frontend CPU flaps (unrelated to incident)
            import math
            metrics["web-frontend"].cpu_usage = 65.0 + 8.0 * math.sin(step * 0.8)

        # Collateral damage: wrong rollback of user-service causes brief outage
        if "rollback:user-service" in fixes:
            us = metrics["user-service"]
            if step <= 2:
                # Rollback disruption: user-service offline during rollback
                us.error_rate = min(8.0, us.error_rate + 4.0)
                us.latency_p99 = min(5000, us.latency_p99 + 1200)
            else:
                # Recovers, but api-gateway still broken
                us.error_rate = max(0.1, us.error_rate - 1.5)

    def sla_ok(metrics: dict[str, ServiceMetrics]) -> bool:
        return metrics["api-gateway"].error_rate < 5.0

    return MetricEngine(initial, evolve, sla_ok)


# ---------------------------------------------------------------------------
# Task 3 — Cascading failure metric engine factory
# ---------------------------------------------------------------------------

def make_cascade_metric_engine() -> MetricEngine:
    """
    order-service leaks 85 DB connections/step.
    db-pool exhausts quickly; all DB-dependent services fail.
    Fix: restart order-service (stop leak) + scale db-pool (clear queue).
    """
    initial = {
        "db-pool": ServiceMetrics(
            connections_used=200.0, wait_queue=847.0,
            error_rate=92.0,  # proxy: % of connection requests failing
        ),
        "order-service": ServiceMetrics(
            error_rate=0.0, latency_p99=145, memory_usage=52.0,
            cpu_usage=44.0, requests_per_sec=148.0,
            db_connections_acquired=340.0,
        ),
        "payment-service": ServiceMetrics(
            error_rate=89.2, latency_p99=30000, memory_usage=41.0,
            cpu_usage=22.0, requests_per_sec=4.0,
        ),
        "inventory-service": ServiceMetrics(
            error_rate=91.4, latency_p99=30000, memory_usage=38.0,
            cpu_usage=20.0, requests_per_sec=3.0,
        ),
        "shipping-service": ServiceMetrics(
            error_rate=87.8, latency_p99=30000, memory_usage=35.0,
            cpu_usage=18.0, requests_per_sec=3.0,
        ),
        "web-frontend": ServiceMetrics(
            # Red herring: elevated CPU (unrelated — marketing campaign traffic)
            error_rate=0.08, latency_p99=220, memory_usage=61.0,
            cpu_usage=78.0, requests_per_sec=2100.0,
        ),
        "user-auth-service": ServiceMetrics(
            # New red herring: elevated latency because it also makes DB calls
            # that are slow due to pool exhaustion — looks like a root cause but is a symptom
            error_rate=4.2, latency_p99=890, memory_usage=44.0,
            cpu_usage=31.0, requests_per_sec=182.0,
        ),
    }

    def evolve(metrics: dict[str, ServiceMetrics], step: int, fixes: set[str]) -> None:
        order_fixed = (
            "restart:order-service" in fixes or
            "rollback:order-service" in fixes
        )
        pool_scaled = "scale:db-pool" in fixes

        db = metrics["db-pool"]
        order = metrics["order-service"]

        if order_fixed:
            order.db_connections_acquired = max(45.0, order.db_connections_acquired - 60.0)
            if pool_scaled:
                # Full recovery
                db.wait_queue = max(0, db.wait_queue - 200)
                db.connections_used = max(42, db.connections_used - 40)
                db.error_rate = max(0.0, db.error_rate - 22.0)
                for svc in ["payment-service", "inventory-service", "shipping-service"]:
                    s = metrics[svc]
                    s.error_rate = max(0.1, s.error_rate - 18.0)
                    s.latency_p99 = max(120, s.latency_p99 - 6000)
                    s.requests_per_sec = min(40, s.requests_per_sec + 8)
                # user-auth-service also recovers
                uas = metrics.get("user-auth-service")
                if uas:
                    uas.error_rate = max(0.1, uas.error_rate - 2.0)
                    uas.latency_p99 = max(95, uas.latency_p99 - 200)
            else:
                # Leak stopped but pool still exhausted; slow drain
                db.wait_queue = max(0, db.wait_queue - 15)
                db.error_rate = max(50.0, db.error_rate - 5.0)
        else:
            # Leak continues: pool gets worse each step
            order.db_connections_acquired = min(500.0, order.db_connections_acquired + 12.0)
            db.wait_queue = min(2000, db.wait_queue + 80)
            db.error_rate = min(100.0, db.error_rate + 1.5)
            for svc in ["payment-service", "inventory-service", "shipping-service"]:
                s = metrics[svc]
                s.error_rate = min(99.0, s.error_rate + 0.8)
            # user-auth-service also degrades (another DB-dependent symptom)
            uas = metrics.get("user-auth-service")
            if uas:
                uas.error_rate = min(30.0, uas.error_rate + 0.5)
                uas.latency_p99 = min(30000, uas.latency_p99 + 150)

        # Pool scaled BEFORE order fixed: dramatic re-exhaustion to show ordering matters
        if pool_scaled and not order_fixed:
            # The leak continues to fill the pool back up visibly
            db.connections_used = min(200.0, db.connections_used + 15.0)
            db.wait_queue = min(2000, db.wait_queue + 60)  # net growth after partial drain

        # Restarting downstream services: temporary relief only
        for svc in ["payment-service", "inventory-service", "shipping-service"]:
            fix_key = f"restart:{svc}"
            if fix_key in fixes and not order_fixed:
                s = metrics[svc]
                # Give a 2-step grace then worsen again (handled by error_rate growth above)
                if step <= 3:
                    s.error_rate = max(5.0, s.error_rate - 20.0)

    def sla_ok(metrics: dict[str, ServiceMetrics]) -> bool:
        # SLA: all critical services must have < 5% error rate
        for svc in ["payment-service", "inventory-service", "shipping-service"]:
            if metrics[svc].error_rate >= 5.0:
                return False
        return True

    return MetricEngine(initial, evolve, sla_ok)


# ---------------------------------------------------------------------------
# Task 4 — Config Drift (TLS cert CN mismatch) metric engine factory
# ---------------------------------------------------------------------------

def make_config_drift_metric_engine() -> MetricEngine:
    """
    checkout-service TLS handshake failing because payments-gateway cert was rotated
    to a new CN. Error rate grows 0.8%/step without fix.
    update_config:checkout-service = full recovery.
    restart:checkout-service = temporary ~30% improvement then regresses.
    """
    initial = {
        "checkout-service": ServiceMetrics(
            error_rate=18.2, latency_p99=8200, memory_usage=45.0,
            cpu_usage=32.0, requests_per_sec=180.0,
        ),
        "payments-gateway": ServiceMetrics(
            # Healthy — cert rotation completed successfully on gateway side
            error_rate=0.0, latency_p99=42, memory_usage=38.0,
            cpu_usage=18.0, requests_per_sec=0.0,
        ),
        "inventory-service": ServiceMetrics(
            # Downstream symptom: reduced checkout flow
            error_rate=3.2, latency_p99=890, memory_usage=41.0,
            cpu_usage=28.0, requests_per_sec=85.0,
        ),
        "redis-session": ServiceMetrics(
            # Red herring: session retries from failed checkouts inflate connection count
            error_rate=0.05, latency_p99=12, memory_usage=52.0,
            cpu_usage=24.0, connections_used=2840.0,
        ),
    }

    def evolve(metrics: dict[str, ServiceMetrics], step: int, fixes: set[str]) -> None:
        co = metrics["checkout-service"]
        config_fixed = "update_config:checkout-service" in fixes
        restarted = "restart:checkout-service" in fixes

        if config_fixed:
            # Full recovery: TLS mismatch resolved, error rate drops to near-zero
            co.error_rate = max(0.1, co.error_rate - 9.0)
            co.latency_p99 = max(120, co.latency_p99 - 2000)
            co.requests_per_sec = min(220, co.requests_per_sec + 20)
            metrics["inventory-service"].error_rate = max(0.05, metrics["inventory-service"].error_rate - 1.5)
            metrics["redis-session"].connections_used = max(200.0, metrics["redis-session"].connections_used - 320.0)
        elif restarted:
            # Restart gives temporary improvement (reconnects briefly) then TLS fails again
            if step <= 2:
                co.error_rate = max(12.0, co.error_rate - 3.0)
                co.latency_p99 = max(3000, co.latency_p99 - 1000)
            else:
                # TLS mismatch persists — regresses
                co.error_rate = min(40.0, co.error_rate + 0.8)
                co.latency_p99 = min(30000, co.latency_p99 + 200)
        else:
            # Growing failure as more payment attempts fail
            co.error_rate = min(40.0, co.error_rate + 0.8)
            co.latency_p99 = min(30000, co.latency_p99 + 200)
            co.requests_per_sec = max(80, co.requests_per_sec - 8)
            metrics["inventory-service"].error_rate = min(15.0, metrics["inventory-service"].error_rate + 0.3)
            metrics["redis-session"].connections_used = min(3000.0, metrics["redis-session"].connections_used + 80.0)

    def sla_ok(metrics: dict[str, ServiceMetrics]) -> bool:
        return metrics["checkout-service"].error_rate < 5.0

    return MetricEngine(initial, evolve, sla_ok)


# ---------------------------------------------------------------------------
# Task 5 — Alert Storm (consumer deadlock) metric engine factory
# ---------------------------------------------------------------------------

def make_alert_storm_metric_engine() -> MetricEngine:
    """
    notification-service deadlocked — no messages consumed.
    message-queue memory fills at +1.2%/step. Queue depth grows +18K/step (stored as /1000).
    Producer services error rates all elevated.
    Fix: restart notification-service (stops deadlock) then restart message-queue (clears memory).
    Order matters: fix consumer BEFORE clearing queue.
    analytics-service and cdn-proxy are fully unrelated.
    """
    initial = {
        "message-queue": ServiceMetrics(
            # memory_usage = RabbitMQ heap %, error_rate = % of publish attempts failing
            # wait_queue = queue depth in thousands (displayed as "847 waiting" = 847K messages)
            error_rate=92.0, memory_usage=97.0,
            wait_queue=847.0, connections_used=200.0,
        ),
        "notification-service": ServiceMetrics(
            # Deadlocked internally but HTTP health check passes — looks healthy from outside!
            # db_connections_acquired = processing_rate (0/min = no consumption)
            error_rate=0.0, cpu_usage=45.0,
            db_connections_acquired=0.0,  # processing rate: 0 messages/min
        ),
        "order-service": ServiceMetrics(
            error_rate=34.2, latency_p99=30000, memory_usage=52.0,
            cpu_usage=78.0, requests_per_sec=148.0,
        ),
        "payment-service": ServiceMetrics(
            error_rate=31.8, latency_p99=30000, memory_usage=41.0,
            cpu_usage=62.0, requests_per_sec=38.0,
        ),
        "email-service": ServiceMetrics(
            error_rate=28.4, latency_p99=30000, memory_usage=38.0,
            cpu_usage=55.0, requests_per_sec=240.0,
        ),
        "user-service": ServiceMetrics(
            error_rate=22.1, latency_p99=8200, memory_usage=44.0,
            cpu_usage=48.0, requests_per_sec=412.0,
        ),
        "analytics-service": ServiceMetrics(
            # Unrelated: disk issue from failed log rotation (72h alert duration)
            error_rate=0.0, cpu_usage=40.0, memory_usage=55.0,
        ),
        "cdn-proxy": ServiceMetrics(
            # Unrelated: health check flap from network hiccup
            error_rate=0.02, cpu_usage=12.0, latency_p99=28,
        ),
    }

    def evolve(metrics: dict[str, ServiceMetrics], step: int, fixes: set[str]) -> None:
        notif_fixed = "restart:notification-service" in fixes
        queue_fixed = "restart:message-queue" in fixes
        mq = metrics["message-queue"]
        ns = metrics["notification-service"]

        if notif_fixed:
            # Consumer deadlock cleared — processing resumes, queue drains
            ns.db_connections_acquired = min(1200.0, ns.db_connections_acquired + 150.0)  # processing rate recovering
            mq.wait_queue = max(0, mq.wait_queue - 25)  # queue draining 25K/step
            mq.memory_usage = max(40.0, mq.memory_usage - 0.8)

            if queue_fixed:
                # Full recovery: queue cleared, memory freed, producers recover
                mq.wait_queue = max(0, mq.wait_queue - 150)  # fast drain
                mq.memory_usage = max(20.0, mq.memory_usage - 8.0)
                mq.error_rate = max(0.0, mq.error_rate - 25.0)
                for svc in ["order-service", "payment-service", "email-service", "user-service"]:
                    s = metrics[svc]
                    s.error_rate = max(0.1, s.error_rate - 12.0)
                    s.latency_p99 = max(90, s.latency_p99 - 8000)
                    s.requests_per_sec = min(s.requests_per_sec + 30, s.requests_per_sec + 30)
        elif queue_fixed and not notif_fixed:
            # Wrong order: cleared queue but consumer still deadlocked
            # Queue refills because notification-service still not consuming
            mq.memory_usage = min(99.0, mq.memory_usage + 0.5)  # memory starts climbing again
            mq.wait_queue = min(2000, mq.wait_queue + 15)  # queue refilling
            mq.error_rate = min(99.0, mq.error_rate + 0.8)
        else:
            # No fix: queue fills, memory grows, producers timeout
            mq.memory_usage = min(99.0, mq.memory_usage + 1.2)
            mq.wait_queue = min(2000, mq.wait_queue + 18)  # 18K messages/step
            mq.error_rate = min(99.0, mq.error_rate + 0.5)
            for svc in ["order-service", "payment-service", "email-service", "user-service"]:
                s = metrics[svc]
                s.error_rate = min(99.0, s.error_rate + 0.8)

        # Restarting individual producers: temporary but errors return when queue stays full
        for svc in ["order-service", "payment-service", "email-service", "user-service"]:
            fix_key = f"restart:{svc}"
            if fix_key in fixes and not notif_fixed:
                s = metrics[svc]
                if step <= 2:
                    s.error_rate = max(5.0, s.error_rate - 15.0)  # brief relief

    def sla_ok(metrics: dict[str, ServiceMetrics]) -> bool:
        return metrics["payment-service"].error_rate < 5.0

    return MetricEngine(initial, evolve, sla_ok)
