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
    """
    initial = {
        "payment-service": ServiceMetrics(
            error_rate=3.2, latency_p99=4200, memory_usage=97.0,
            cpu_usage=62.0, requests_per_sec=38.0,
        ),
        "api-gateway": ServiceMetrics(
            error_rate=0.02, latency_p99=95, memory_usage=41.0,
            cpu_usage=28.0, requests_per_sec=420.0,
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
