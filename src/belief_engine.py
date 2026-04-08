"""
belief_engine.py
================
Epistemic Bayesian Reward Signal — per-episode belief distribution over root cause hypotheses.

Each investigation action updates the distribution using pre-specified likelihood weights.
The per-step reward is Shannon entropy reduction: how much the agent's uncertainty sharpened.

Design
------
- Hypothesis space: a fixed set of competing root cause hypotheses per incident
- Prior: uniform (agent knows nothing at episode start beyond the alert list)
- Likelihood table: P(seeing tool+service | hypothesis is true) — hand-authored per incident
- Bayesian update: unnormalized multiply then renormalize after each tool call
- Redundant query: same tool:service called twice gets 90% shrunk likelihood delta → near-zero gain
- Epistemic gain: (H_prior - H_posterior) / log2(N), scaled to ~0.01–0.04 per step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BeliefState:
    """Snapshot of belief distribution at one timestep."""
    probs: dict[str, float]
    top_hypothesis: str
    confidence: float
    entropy: float


class BeliefEngine:
    """
    Maintains a probability distribution over root cause hypotheses.

    Parameters
    ----------
    hypothesis_space : list[str]
        All hypotheses the agent could believe are the root cause.
        e.g. ["oom:payment-service", "bad_deploy:payment-service", "red_herring:api-gateway-cpu"]
    likelihood_table : dict[str, dict[str, float]]
        Maps hypothesis -> {tool_key -> likelihood weight}.
        tool_key = "tool:service_or_topic" (e.g. "query_logs:payment-service").
        Values in [0, 1]. Default for unspecified keys is 0.5 (uninformative).
    true_hypothesis : str
        The actual root cause. Hidden from the agent. Used only in final scoring.
    """

    def __init__(
        self,
        hypothesis_space: list[str],
        likelihood_table: dict[str, dict[str, float]],
        true_hypothesis: str,
    ) -> None:
        self.hypothesis_space = hypothesis_space
        self.likelihood_table = likelihood_table
        self.true_hypothesis = true_hypothesis

        n = len(hypothesis_space)
        self._probs: dict[str, float] = {h: 1.0 / n for h in hypothesis_space}
        self._called_keys: set[str] = set()
        self._epistemic_gains: list[float] = []

        # Track whether the agent tried to remediate before being confident
        self.premature_remediation_count: int = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _entropy(self, probs: Optional[dict[str, float]] = None) -> float:
        """Shannon entropy in bits."""
        p = probs or self._probs
        h = 0.0
        for v in p.values():
            if v > 1e-10:
                h -= v * math.log2(v)
        return h

    def _make_tool_key(self, tool: str, args: dict) -> str:
        service = args.get("service", args.get("target", ""))
        topic = args.get("topic", "")
        label = (service or topic or "unknown").lower().strip()
        return f"{tool}:{label}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tool: str, args: dict) -> float:
        """
        Bayesian update on one tool call.

        Returns the epistemic gain (entropy reduction, scaled to [0, 0.04]).
        Call this for: query_logs, query_metrics, read_runbook, check_deploy_history.
        """
        tool_key = self._make_tool_key(tool, args)

        h_prior = self._entropy()

        # Collect likelihoods; shrink by 90% for repeated queries (redundancy penalty)
        redundant = tool_key in self._called_keys
        lh: dict[str, float] = {}
        for h in self.hypothesis_space:
            base = self.likelihood_table.get(h, {}).get(tool_key, 0.5)
            if redundant:
                # Shrink the deviation from 0.5 so repeated queries carry almost no signal
                base = 0.5 + (base - 0.5) * 0.1
            lh[h] = base

        self._called_keys.add(tool_key)

        # Bayesian multiply + renormalize
        unnorm = {h: self._probs[h] * lh[h] for h in self.hypothesis_space}
        total = sum(unnorm.values())
        if total < 1e-12:
            # Degenerate case — leave distribution unchanged
            self._epistemic_gains.append(0.0)
            return 0.0

        self._probs = {h: v / total for h, v in unnorm.items()}

        h_posterior = self._entropy()

        # Normalize by maximum possible entropy (log2 N for uniform over N hypotheses)
        n = len(self.hypothesis_space)
        max_entropy = math.log2(n) if n > 1 else 1.0
        delta_normalized = max(0.0, (h_prior - h_posterior) / max_entropy)

        # Scale to reward magnitude ~0.01–0.04
        scaled = round(delta_normalized * 0.04, 5)
        self._epistemic_gains.append(scaled)
        return scaled

    def notify_remediation(self) -> None:
        """
        Call before an execute_remediation action to check premature remediation.
        Increments premature_remediation_count if not yet confident.
        """
        if not self.is_confident_enough():
            self.premature_remediation_count += 1

    def confidence(self) -> float:
        """Probability mass on the current top hypothesis."""
        return max(self._probs.values())

    def top_hypothesis(self) -> str:
        return max(self._probs, key=lambda h: self._probs[h])

    def is_confident_enough(self, threshold: float = 0.65) -> bool:
        return self.confidence() >= threshold

    def final_confidence_on_true(self) -> float:
        """Probability mass on the true hypothesis at episode end."""
        return self._probs.get(self.true_hypothesis, 0.0)

    def cumulative_gain(self) -> float:
        """Sum of all per-step epistemic gains."""
        return sum(self._epistemic_gains)

    def redundancy_ratio(self) -> float:
        """Fraction of investigative queries that returned near-zero epistemic gain."""
        if not self._epistemic_gains:
            return 0.0
        near_zero = sum(1 for g in self._epistemic_gains if g < 0.001)
        return near_zero / len(self._epistemic_gains)

    def snapshot(self) -> BeliefState:
        top = self.top_hypothesis()
        return BeliefState(
            probs=dict(self._probs),
            top_hypothesis=top,
            confidence=self._probs[top],
            entropy=self._entropy(),
        )


# ---------------------------------------------------------------------------
# Per-task likelihood tables
# ---------------------------------------------------------------------------
# Likelihood value semantics:
#   0.9+ = very strong evidence under this hypothesis
#   0.7  = strong evidence
#   0.5  = uninformative / neutral (default)
#   0.3  = mild evidence against
#   0.1  = strong evidence against (this hypothesis is unlikely given this query)
# ---------------------------------------------------------------------------

# Task 1 — OOM
# Hypotheses: oom:payment-service (TRUE), bad_deploy:payment-service, red_herring:api-gateway-cpu

OOM_HYPOTHESIS_SPACE = [
    "oom:payment-service",
    "bad_deploy:payment-service",
    "red_herring:api-gateway-cpu",
]

OOM_TRUE_HYPOTHESIS = "oom:payment-service"

OOM_LIKELIHOOD_TABLE: dict[str, dict[str, float]] = {
    "oom:payment-service": {
        "query_logs:payment-service":        0.90,  # OOM stacktrace is right there
        "query_metrics:payment-service":     0.85,  # memory 97% confirms heap exhaustion
        "read_runbook:payment-service":      0.70,  # runbook calls OOM the most common issue
        "read_runbook:oom":                  0.90,  # directly about OOM
        "read_runbook:memory":               0.80,  # memory runbook
        "check_deploy_history:payment-service": 0.30,  # deploy is fine; no signal for OOM
        "query_logs:api-gateway":            0.10,  # api-gateway is unrelated
        "query_metrics:api-gateway":         0.10,
        "check_deploy_history:api-gateway":  0.10,
    },
    "bad_deploy:payment-service": {
        "query_logs:payment-service":        0.45,  # logs exist but no deploy error signal
        "query_metrics:payment-service":     0.35,  # metrics don't point to deploy
        "read_runbook:payment-service":      0.50,  # neutral
        "read_runbook:oom":                  0.20,  # wrong runbook for deploy
        "read_runbook:memory":               0.20,
        "check_deploy_history:payment-service": 0.75,  # would show recent v3.8.2 deploy
        "query_logs:api-gateway":            0.15,
        "query_metrics:api-gateway":         0.15,
        "check_deploy_history:api-gateway":  0.15,
    },
    "red_herring:api-gateway-cpu": {
        "query_logs:payment-service":        0.10,  # not about api-gateway
        "query_metrics:payment-service":     0.10,
        "read_runbook:payment-service":      0.15,
        "read_runbook:oom":                  0.10,
        "read_runbook:memory":               0.10,
        "check_deploy_history:payment-service": 0.10,
        "query_logs:api-gateway":            0.85,  # CPU logs would show up here
        "query_metrics:api-gateway":         0.80,  # CPU metric confirms
        "check_deploy_history:api-gateway":  0.50,  # neutral
        "read_runbook:api-gateway":          0.80,  # api-gateway runbook
    },
}

# Task 2 — Bad Deploy
# Hypotheses: bad_deploy:api-gateway-v2.3.1 (TRUE), bad_deploy:user-service-v1.5.0, red_herring:web-frontend-cpu

DEPLOY_HYPOTHESIS_SPACE = [
    "bad_deploy:api-gateway-v2.3.1",
    "bad_deploy:user-service-v1.5.0",
    "red_herring:web-frontend-cpu",
]

DEPLOY_TRUE_HYPOTHESIS = "bad_deploy:api-gateway-v2.3.1"

DEPLOY_LIKELIHOOD_TABLE: dict[str, dict[str, float]] = {
    "bad_deploy:api-gateway-v2.3.1": {
        "query_logs:api-gateway":            0.90,  # NPE logs directly point to v2.3.1
        "check_deploy_history:api-gateway":  0.88,  # shows v2.3.1 deployed 5min ago
        "query_metrics:api-gateway":         0.80,  # error_rate spike at deploy time
        "read_runbook:api-gateway":          0.65,  # somewhat relevant
        "read_runbook:rollback":             0.80,  # rollback runbook = strong signal
        "query_logs:user-service":           0.20,  # user-service is a red herring
        "check_deploy_history:user-service": 0.25,  # shows older deploy, not correlated
        "query_metrics:user-service":        0.25,
        "query_logs:web-frontend":           0.10,
        "query_metrics:web-frontend":        0.10,
        "check_deploy_history:web-frontend": 0.10,
    },
    "bad_deploy:user-service-v1.5.0": {
        "query_logs:api-gateway":            0.25,  # api-gateway errors don't match user-service deploy
        "check_deploy_history:api-gateway":  0.30,  # timing mismatch with error spike
        "query_metrics:api-gateway":         0.30,  # error at gateway, not user-service
        "read_runbook:api-gateway":          0.35,
        "read_runbook:rollback":             0.60,
        "query_logs:user-service":           0.70,  # logs would show if user-svc had issues
        "check_deploy_history:user-service": 0.80,  # 44min old deploy visible
        "query_metrics:user-service":        0.65,
        "query_logs:web-frontend":           0.15,
        "query_metrics:web-frontend":        0.15,
        "check_deploy_history:web-frontend": 0.10,
    },
    "red_herring:web-frontend-cpu": {
        "query_logs:api-gateway":            0.10,
        "check_deploy_history:api-gateway":  0.10,
        "query_metrics:api-gateway":         0.10,
        "read_runbook:api-gateway":          0.10,
        "read_runbook:rollback":             0.20,
        "query_logs:user-service":           0.15,
        "check_deploy_history:user-service": 0.15,
        "query_metrics:user-service":        0.15,
        "query_logs:web-frontend":           0.85,  # CPU logs for marketing campaign
        "query_metrics:web-frontend":        0.80,  # CPU high
        "check_deploy_history:web-frontend": 0.50,
    },
}

# Task 3 — Cascade (Connection Leak)
# Hypotheses: connection_leak:order-service (TRUE), symptom:payment-service,
#             red_herring:web-frontend, red_herring:user-auth-service (new)
# NOTE: payment-service is removed from red_herring (was a grader bug).
#       user-auth-service added as a new, subtler red herring.

CASCADE_HYPOTHESIS_SPACE = [
    "connection_leak:order-service",
    "symptom:payment-service",
    "red_herring:web-frontend",
    "red_herring:user-auth-service",
]

CASCADE_TRUE_HYPOTHESIS = "connection_leak:order-service"

CASCADE_LIKELIHOOD_TABLE: dict[str, dict[str, float]] = {
    "connection_leak:order-service": {
        "query_logs:db-pool":                    0.90,  # pool exhaustion with leak attribution
        "query_metrics:db-pool":                 0.88,  # 200/200 + growing wait_queue
        "query_logs:order-service":              0.88,  # connection acquisition logs
        "query_metrics:order-service":           0.85,  # db_connections_acquired spike
        "check_deploy_history:order-service":    0.85,  # v4.2.0 at time of incident
        "read_runbook:cascade":                  0.75,
        "read_runbook:db-pool":                  0.70,
        "query_logs:payment-service":            0.40,  # symptom visible but not cause
        "query_metrics:payment-service":         0.35,
        "query_logs:user-auth-service":          0.40,  # also a symptom
        "query_metrics:user-auth-service":       0.35,
        "query_logs:web-frontend":               0.10,
        "query_metrics:web-frontend":            0.10,
    },
    "symptom:payment-service": {
        "query_logs:db-pool":                    0.40,
        "query_metrics:db-pool":                 0.35,
        "query_logs:order-service":              0.20,
        "query_metrics:order-service":           0.20,
        "check_deploy_history:order-service":    0.20,
        "read_runbook:cascade":                  0.50,
        "read_runbook:db-pool":                  0.40,
        "query_logs:payment-service":            0.85,  # payment errors visible
        "query_metrics:payment-service":         0.80,  # high error rate
        "check_deploy_history:payment-service":  0.60,  # 2-day-old deploy suspicious
        "query_logs:user-auth-service":          0.35,
        "query_metrics:user-auth-service":       0.30,
        "query_logs:web-frontend":               0.15,
        "query_metrics:web-frontend":            0.10,
    },
    "red_herring:web-frontend": {
        "query_logs:db-pool":                    0.10,
        "query_metrics:db-pool":                 0.10,
        "query_logs:order-service":              0.10,
        "query_metrics:order-service":           0.10,
        "check_deploy_history:order-service":    0.10,
        "read_runbook:cascade":                  0.15,
        "read_runbook:db-pool":                  0.10,
        "query_logs:payment-service":            0.15,
        "query_metrics:payment-service":         0.10,
        "query_logs:user-auth-service":          0.15,
        "query_metrics:user-auth-service":       0.12,
        "query_logs:web-frontend":               0.88,  # CPU logs for marketing campaign
        "query_metrics:web-frontend":            0.85,
        "check_deploy_history:web-frontend":     0.50,
    },
    "red_herring:user-auth-service": {
        "query_logs:db-pool":                    0.30,  # pool exhaustion visible
        "query_metrics:db-pool":                 0.28,
        "query_logs:order-service":              0.15,
        "query_metrics:order-service":           0.15,
        "check_deploy_history:order-service":    0.15,
        "read_runbook:cascade":                  0.35,
        "read_runbook:db-pool":                  0.30,
        "query_logs:payment-service":            0.30,
        "query_metrics:payment-service":         0.25,
        "query_logs:user-auth-service":          0.78,  # elevated latency visible — looks like root cause
        "query_metrics:user-auth-service":       0.75,  # p99=890ms looks suspicious
        "check_deploy_history:user-auth-service": 0.30,
        "query_logs:web-frontend":               0.15,
        "query_metrics:web-frontend":            0.12,
    },
}


# ---------------------------------------------------------------------------
# Task 4 — Config Drift (TLS cert CN mismatch)
# Hypotheses: config_drift:tls_cert_mismatch (TRUE), code_bug:checkout-service,
#             red_herring:redis-session, red_herring:payments-gateway-overload

CONFIG_DRIFT_HYPOTHESIS_SPACE = [
    "config_drift:tls_cert_mismatch",
    "code_bug:checkout-service",
    "red_herring:redis-session",
    "red_herring:payments-gateway-overload",
]

CONFIG_DRIFT_TRUE_HYPOTHESIS = "config_drift:tls_cert_mismatch"

CONFIG_DRIFT_LIKELIHOOD_TABLE: dict[str, dict[str, float]] = {
    "config_drift:tls_cert_mismatch": {
        "query_logs:checkout-service":           0.88,  # CN mismatch SSLHandshakeException
        "query_logs:payments-gateway":           0.92,  # cert rotation logged at 09:12
        "check_deploy_history:payments-gateway": 0.95,  # infra event: cert_rotation visible
        "read_runbook:tls":                      0.78,
        "read_runbook:checkout-service":         0.75,
        "query_metrics:checkout-service":        0.65,  # high error rate confirms issue
        "query_metrics:payments-gateway":        0.60,  # gateway healthy = cert rotation not a gateway bug
        "check_deploy_history:checkout-service": 0.20,  # no recent deploy STRONGLY disconfirms code_bug
        "query_logs:redis-session":              0.25,  # runbook says it's symptom not cause
        "query_metrics:redis-session":           0.28,
        "read_runbook:redis-session":            0.20,  # runbook explicitly says it's upstream symptom
    },
    "code_bug:checkout-service": {
        "query_logs:checkout-service":           0.55,  # errors present but nothing code-specific
        "check_deploy_history:checkout-service": 0.20,  # NO RECENT DEPLOY = strongly disconfirms
        "query_metrics:checkout-service":        0.60,
        "read_runbook:checkout-service":         0.55,
        "query_logs:payments-gateway":           0.30,  # gateway fine, not relevant to code bug
        "check_deploy_history:payments-gateway": 0.30,
        "read_runbook:tls":                      0.40,  # slightly informative
        "query_logs:redis-session":              0.40,
        "query_metrics:redis-session":           0.40,
    },
    "red_herring:redis-session": {
        "query_logs:redis-session":              0.55,  # elevated connections look suspicious
        "query_metrics:redis-session":           0.60,  # 2840 connections visible
        "read_runbook:redis-session":            0.25,  # runbook says it's a symptom → disconfirms
        "query_logs:checkout-service":           0.30,  # checkout errors exist but point elsewhere
        "query_metrics:checkout-service":        0.35,
        "query_logs:payments-gateway":           0.15,
        "check_deploy_history:payments-gateway": 0.15,
    },
    "red_herring:payments-gateway-overload": {
        "query_metrics:payments-gateway":        0.30,  # metrics look healthy → disconfirms
        "query_logs:payments-gateway":           0.35,  # shows cert rotation, not overload
        "check_deploy_history:payments-gateway": 0.40,  # shows cert rotation (informative, not overload)
        "query_logs:checkout-service":           0.35,
        "read_runbook:tls":                      0.30,
    },
}


# ---------------------------------------------------------------------------
# Task 5 — Alert Storm (Consumer Deadlock)
# Hypotheses: consumer_deadlock:notification-service (TRUE),
#             queue_overload:message-queue, symptom:producer-services,
#             red_herring:analytics-service, red_herring:cdn-proxy

ALERT_STORM_HYPOTHESIS_SPACE = [
    "consumer_deadlock:notification-service",
    "queue_overload:message-queue",
    "symptom:producer-services",
    "red_herring:analytics-service",
    "red_herring:cdn-proxy",
]

ALERT_STORM_TRUE_HYPOTHESIS = "consumer_deadlock:notification-service"

ALERT_STORM_LIKELIHOOD_TABLE: dict[str, dict[str, float]] = {
    "consumer_deadlock:notification-service": {
        "query_logs:message-queue":                  0.82,  # consumer lag + idle consumer visible
        "query_logs:notification-service":           0.95,  # DEADLOCK stacktrace — smoking gun
        "query_metrics:message-queue":               0.85,  # memory 97%, queue depth 847K
        "check_deploy_history:notification-service": 0.90,  # v2.8.0 deployed 3h ago
        "read_runbook:message-queue":                0.80,  # explains consumer-first fix
        "read_runbook:notification-service":         0.85,  # known v2.8.0 deadlock
        "read_runbook:alert-storm":                  0.72,
        "query_logs:order-service":                  0.45,  # queue publish timeout (symptom)
        "query_logs:payment-service":                0.45,  # same pattern
        "check_deploy_history:message-queue":        0.30,  # no changes → points to consumer
        "query_logs:analytics-service":              0.15,
        "query_logs:cdn-proxy":                      0.12,
    },
    "queue_overload:message-queue": {
        "query_logs:message-queue":                  0.65,  # queue depth visible
        "query_metrics:message-queue":               0.70,  # memory and depth
        "query_logs:notification-service":           0.50,  # deadlock present but harder to link
        "check_deploy_history:message-queue":        0.20,  # no changes for 30 days → disconfirms
        "read_runbook:message-queue":                0.60,
        "query_logs:order-service":                  0.55,
        "check_deploy_history:notification-service": 0.55,
        "query_logs:analytics-service":              0.15,
    },
    "symptom:producer-services": {
        "query_logs:order-service":                  0.60,  # queue publish timeout visible
        "query_logs:payment-service":                0.58,
        "query_metrics:order-service":               0.55,
        "check_deploy_history:order-service":        0.30,  # stable 4+ days → disconfirms
        "query_logs:message-queue":                  0.50,
        "query_logs:notification-service":           0.40,
        "check_deploy_history:notification-service": 0.50,
        "query_logs:analytics-service":              0.20,
        "query_logs:cdn-proxy":                      0.15,
    },
    "red_herring:analytics-service": {
        "query_logs:analytics-service":              0.30,  # logs say "unrelated" → strong disconfirm
        "query_metrics:analytics-service":           0.28,
        "check_deploy_history:analytics-service":    0.20,
        "query_logs:message-queue":                  0.20,
        "query_logs:notification-service":           0.15,
        "query_logs:order-service":                  0.25,
    },
    "red_herring:cdn-proxy": {
        "query_logs:cdn-proxy":                      0.25,  # health check flap, explicitly unrelated
        "query_metrics:cdn-proxy":                   0.22,
        "query_logs:message-queue":                  0.15,
        "query_logs:notification-service":           0.12,
        "query_logs:order-service":                  0.18,
    },
}
