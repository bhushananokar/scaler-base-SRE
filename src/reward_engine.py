"""
reward_engine.py
================
8-dimensional reward system with 29 individual signal components for
incident response agent training.

Architecture
------------
- Per-step rewards  (~30% of total): small immediate signal on each action
- Episode-end rewards (~70% of total): computed at declare_resolved
- Total normalised to 0.0–1.0

Dimensions
----------
D1  Situational Awareness   (alert ack, blast radius, investigation depth)
D2  Diagnostic Quality      (root cause ID, efficiency, coherence, red-herring resistance)
D3  Remediation Quality     (action correctness, ordering, collateral damage, fix verification)
D4  Time Efficiency         (MTTD, MTTR, SLA compliance)
D5  Communication           (update cadence, update quality, resolution accuracy)
D6  Anti-pattern Penalties  (blind remediations, circular queries, red herring actions, unresolved)
D7  Epistemic Quality       (cumulative entropy reduction, final belief confidence, redundancy)
D8  Workflow Coherence      (phase progression, verify-before-resolve bonus)

No external dependencies — standard library + dataclasses + typing only.
BeliefEngine and WorkflowMachine are passed into compute_final_reward at episode end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .belief_engine import BeliefEngine
    from .workflow_machine import WorkflowMachine


# ---------------------------------------------------------------------------
# IncidentConfig
# ---------------------------------------------------------------------------

@dataclass
class IncidentConfig:
    """Static description of one incident scenario used by RewardEngine."""

    task_id: int
    severity_level: str                      # "SEV1", "SEV2", "SEV3"

    root_cause_service: str                  # e.g. "payment-service"
    root_cause_type: str                     # "oom", "bad_deploy", "connection_leak"
    root_cause_keywords: list[str]           # expected in declare_resolved root_cause string

    all_services: list[str]                  # every service in the incident
    relevant_services: list[str]             # services that matter for diagnosis

    red_herring_services: list[str]          # services with misleading signals

    # Golden path: list of (action_type, target) tuples
    golden_actions: list[tuple[str, str]]
    # Ordering constraints: list of (i, j) meaning golden_actions[i] must come before golden_actions[j]
    action_order_constraints: list[tuple[int, int]]

    # Per-dimension weight budget (8 keys required)
    # Keys: situational, diagnostic, remediation, time, communication, anti_patterns,
    #       epistemic_quality, workflow_coherence
    weights: dict[str, float]

    # Which service + metric defines business SLA
    sla_service: str
    sla_metric: str                          # "error_rate", "latency_p99", etc.

    # Epistemic fields (populated per-task)
    hypothesis_space: list[str] = field(default_factory=list)
    likelihood_table: dict[str, dict[str, float]] = field(default_factory=dict)
    true_hypothesis: str = ""

    # Episode length cap (default 15; harder tasks may use 20)
    max_steps: int = 15


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Stores per-dimension scores and line-item components from one episode."""

    # Dimension scores (raw, before clamping)
    situational: float = 0.0
    diagnostic: float = 0.0
    remediation: float = 0.0
    time_efficiency: float = 0.0
    communication: float = 0.0
    anti_patterns: float = 0.0      # will be <= 0
    epistemic_quality: float = 0.0
    workflow_coherence: float = 0.0

    # Line-item detail
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)

    # Human-readable narrative hints set by the engine
    _narrative_hints: list[str] = field(default_factory=list, repr=False)

    @property
    def total(self) -> float:
        """Sum of all dimension scores, clamped to [0, 1]."""
        raw = (
            self.situational
            + self.diagnostic
            + self.remediation
            + self.time_efficiency
            + self.communication
            + self.anti_patterns   # negative
            + self.epistemic_quality
            + self.workflow_coherence
        )
        return max(0.0, min(1.0, raw))

    def to_feedback(self) -> str:
        """Return a multi-line human-readable breakdown string."""
        lines: list[str] = []
        lines.append("=== INCIDENT RESOLVED ===")
        lines.append(f"Final Score: {self.total:.3f}/1.000")
        lines.append("")

        # Helper to look up component values safely
        def c(name: str) -> float:
            return self.components.get(name, 0.0)

        def p(name: str) -> float:
            return self.penalties.get(name, 0.0)

        # ---- D1 ----
        d1_label = "Situational Awareness"
        d1_max = self.components.get("_d1_max", 0.0)
        lines.append(f"[D1] {d1_label}:     {self.situational:.3f} / {d1_max:.3f}")

        aa = c("alert_ack")
        br = c("blast_radius")
        idepth = c("investigation_depth")
        lines.append(f"  alert_ack:              +{aa:.3f}  (acknowledged alerts at step "
                     f"{self.components.get('_alert_ack_step', '?')})")
        lines.append(f"  blast_radius:           +{br:.3f}  "
                     f"(investigated {self.components.get('_relevant_investigated', '?')}/"
                     f"{self.components.get('_relevant_total', '?')} relevant services)")
        lines.append(f"  investigation_depth:    +{idepth:.3f}  "
                     f"(relevant query ratio: {self.components.get('_rel_ratio', 0.0):.2f})")
        lines.append("")

        # ---- D2 ----
        d2_max = self.components.get("_d2_max", 0.0)
        lines.append(f"[D2] Diagnostic Quality:        {self.diagnostic:.3f} / {d2_max:.3f}")
        lines.append(f"  root_cause_id:          +{c('root_cause_id'):.3f}  "
                     f"({int(self.components.get('_kw_matches', 0))} root cause keywords matched)")
        lines.append(f"  investigation_efficiency: +{c('investigation_efficiency'):.3f}  "
                     f"(relevance ratio: {self.components.get('_inv_eff_ratio', 0.0):.2f})")
        lines.append(f"  hypothesis_coherence:   +{c('hypothesis_coherence'):.3f}  "
                     f"({'root cause service investigated before fix' if c('hypothesis_coherence') > 0 else 'root cause service NOT investigated before fix'})")
        lines.append(f"  red_herring_resistance: +{c('red_herring_resistance'):.3f}  "
                     f"({int(self.components.get('_rh_actions', 0))} red herring action(s) taken)")
        lines.append("")

        # ---- D3 ----
        d3_max = self.components.get("_d3_max", 0.0)
        lines.append(f"[D3] Remediation Quality:       {self.remediation:.3f} / {d3_max:.3f}")
        lines.append(f"  action_correctness:     +{c('action_correctness'):.3f}  "
                     f"({int(self.components.get('_correct_applied', 0))}/"
                     f"{int(self.components.get('_golden_total', 0))} golden actions applied)")
        lines.append(f"  action_ordering:        +{c('action_ordering'):.3f}  "
                     f"({'all ordering constraints satisfied' if c('action_ordering') == self.components.get('_ord_max', 0.0) else 'some ordering constraints violated'})")
        lines.append(f"  collateral_damage:      +{c('collateral_damage'):.3f}  "
                     f"({int(self.components.get('_wrong_actions', 0))} wrong action(s))")
        lines.append(f"  fix_verification:       +{c('fix_verification'):.3f}  "
                     f"({'metrics checked after fix' if c('fix_verification') > 0 else 'metrics NOT checked after fix'})")
        lines.append("")

        # ---- D4 ----
        d4_max = self.components.get("_d4_max", 0.0)
        lines.append(f"[D4] Time Efficiency:           {self.time_efficiency:.3f} / {d4_max:.3f}")
        first_rel = self.components.get('_first_rel_step', None)
        first_rel_str = f"step {first_rel}" if first_rel is not None else "never"
        lines.append(f"  mttd:                   +{c('mttd'):.3f}  "
                     f"(first relevant investigation at {first_rel_str})")
        rem_delay = self.components.get('_rem_delay', None)
        rem_delay_str = f"{rem_delay} steps after diagnosis" if rem_delay is not None else "N/A"
        lines.append(f"  mttr:                   +{c('mttr'):.3f}  "
                     f"(fix applied {rem_delay_str})")
        sla_viol = int(self.components.get('_sla_viol_steps', 0))
        total_st = int(self.components.get('_total_steps', 1))
        lines.append(f"  sla_compliance:         +{c('sla_compliance'):.3f}  "
                     f"(SLA violated {sla_viol}/{total_st} steps)")
        lines.append("")

        # ---- D5 ----
        d5_max = self.components.get("_d5_max", 0.0)
        lines.append(f"[D5] Communication:             {self.communication:.3f} / {d5_max:.3f}")
        n_updates = int(self.components.get('_n_updates', 0))
        lines.append(f"  update_cadence:         +{c('update_cadence'):.3f}  "
                     f"({n_updates} status update(s) posted)")
        lines.append(f"  update_quality:         +{c('update_quality'):.3f}  "
                     f"(updates mentioned relevant services)")
        lines.append(f"  resolution_accuracy:    +{c('resolution_accuracy'):.3f}  "
                     f"({'root cause was specific' if c('resolution_accuracy') > 0 else 'root cause was empty or too short'})")
        lines.append("")

        # ---- D6 ----
        lines.append(f"[D6] Anti-pattern Penalties:   {self.anti_patterns:.3f}")
        lines.append(f"  blind_remediations:     {p('blind_remediations'):.3f}")
        lines.append(f"  circular_queries:       {p('circular_queries'):.3f}")
        lines.append(f"  red_herring_actions:    {p('red_herring_actions'):.3f}")
        lines.append(f"  unresolved_penalty:     {p('unresolved_penalty'):.3f}")
        lines.append("")

        # ---- D7 ----
        d7_max = self.components.get("_d7_max", 0.0)
        lines.append(f"[D7] Epistemic Quality:         {self.epistemic_quality:.3f} / {d7_max:.3f}")
        lines.append(f"  cumulative_epistemic_gain: +{c('cumulative_epistemic_gain'):.3f}  "
                     f"(total entropy reduction: {self.components.get('_cum_gain_raw', 0.0):.4f})")
        lines.append(f"  final_belief_confidence: +{c('final_belief_confidence'):.3f}  "
                     f"(probability on true hypothesis: {self.components.get('_true_conf', 0.0):.2f})")
        lines.append(f"  redundancy_penalty:     +{c('redundancy_penalty'):.3f}  "
                     f"(redundant query ratio: {self.components.get('_redundancy_ratio', 0.0):.2f})")
        lines.append("")

        # ---- D8 ----
        d8_max = self.components.get("_d8_max", 0.0)
        lines.append(f"[D8] Workflow Coherence:        {self.workflow_coherence:.3f} / {d8_max:.3f}")
        lines.append(f"  phase_progression:      +{c('phase_progression'):.3f}  "
                     f"(phases: {self.components.get('_phase_summary', '?')})")
        lines.append(f"  verify_bonus:           +{c('verify_bonus'):.3f}  "
                     f"({'verified before resolve' if c('verify_bonus') > 0 else 'did NOT verify before resolve'})")
        lines.append("")

        # Narrative hints
        for hint in self._narrative_hints:
            lines.append(hint)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RewardEngine
# ---------------------------------------------------------------------------

class RewardEngine:
    """
    Computes per-step and episode-end rewards for an incident response agent.

    Usage
    -----
    engine = RewardEngine(config)
    # On each agent action:
    delta = engine.on_action(tool, args, step_number)
    # Each step the SLA is violated:
    engine.on_sla_violation_step()
    # After a correct remediation, when metrics are re-queried:
    engine.mark_fix_verified()
    # At episode end:
    score, breakdown = engine.compute_final_reward(root_cause_str, total_steps,
                                                    belief_engine, workflow_machine)
    """

    def __init__(self, config: IncidentConfig) -> None:
        self.config = config

        # ---- Per-step tracking ----
        self.step_count: int = 0
        self.cumulative_step_reward: float = 0.0

        # Alert acknowledgement
        self.alert_ack_step: Optional[int] = None

        # Investigation tracking
        self.services_investigated: set[str] = set()
        self.relevant_services_investigated: set[str] = set()
        self.investigation_queries: list[tuple[str, dict]] = []  # (tool, args)
        self.last_query_key: Optional[str] = None
        self.circular_query_count: int = 0
        self.investigation_steps: int = 0

        # Remediation tracking
        self.remediations_before_investigation: int = 0
        self.actions_taken: list[tuple[str, str]] = []
        self.correct_actions_applied: set[int] = set()  # indices into golden_actions
        self.wrong_actions_count: int = 0
        self.fix_verified: bool = False
        self.first_correct_action_step: Optional[int] = None

        # Red herring
        self.red_herring_actions: int = 0

        # Communication
        self.status_updates: list[tuple[str, int]] = []  # (message, step)

        # SLA / timing
        self.sla_violation_steps: int = 0
        self.total_steps: int = 0

        # Internal: track first step a relevant service was investigated
        self._first_relevant_inv_step: Optional[int] = None

        # Internal: flag set during on_action for WorkflowMachine
        self._last_was_correct_action: bool = False

    # -----------------------------------------------------------------------
    # Per-step methods
    # -----------------------------------------------------------------------

    def on_action(self, tool: str, args: dict, step_number: int) -> float:
        """
        Called on EVERY agent step before the incident environment processes it.

        Returns the per-step reward delta (positive or negative).
        Accumulates into cumulative_step_reward.
        """
        self.total_steps += 1
        self.step_count += 1
        self._last_was_correct_action = False
        delta: float = 0.0

        # ---- list_alerts ----
        if tool == "list_alerts":
            if self.alert_ack_step is None:
                self.alert_ack_step = step_number
            if step_number <= 1:
                delta = 0.03   # fast acknowledgement bonus
            elif step_number <= 3:
                delta = 0.015  # acceptable acknowledgement
            else:
                delta = 0.0

        # ---- query_logs / query_metrics ----
        elif tool in ("query_logs", "query_metrics"):
            self.investigation_steps += 1
            service = _extract_service(args)
            self.services_investigated.add(service)
            self.investigation_queries.append((tool, dict(args)))

            # Circular query detection
            query_key = f"{tool}:{service}"
            if query_key == self.last_query_key:
                self.circular_query_count += 1
                delta = -0.01
                self.last_query_key = query_key
                self.cumulative_step_reward += delta
                return delta
            self.last_query_key = query_key

            if service in self.config.relevant_services:
                self.relevant_services_investigated.add(service)
                # Track when we first investigated a relevant service
                if self._first_relevant_inv_step is None:
                    self._first_relevant_inv_step = step_number
                delta = 0.02
            elif service in self.config.red_herring_services:
                delta = -0.005  # slight negative for investigating noise
            else:
                delta = 0.005   # mild positive for any investigation

        # ---- read_runbook / check_deploy_history ----
        elif tool in ("read_runbook", "check_deploy_history"):
            self.investigation_steps += 1
            delta = 0.01

        # ---- execute_remediation ----
        elif tool == "execute_remediation":
            action_type = args.get("action_type", args.get("type", ""))
            target = args.get("target", args.get("service", ""))

            # Blind remediation (no investigation done yet)
            if self.investigation_steps == 0:
                self.remediations_before_investigation += 1
                delta = -0.03
                self.cumulative_step_reward += delta
                return delta

            self.actions_taken.append((action_type, target))

            # Check against golden actions (fuzzy match on target)
            matched = False
            for idx, (g_type, g_target) in enumerate(self.config.golden_actions):
                if idx in self.correct_actions_applied:
                    continue
                target_match = (
                    target in g_target
                    or g_target in target
                    or target.lower() == g_target.lower()
                )
                if target_match:
                    self.correct_actions_applied.add(idx)
                    if self.first_correct_action_step is None:
                        self.first_correct_action_step = step_number
                    matched = True
                    self._last_was_correct_action = True
                    delta = 0.02
                    break

            if not matched:
                if target in self.config.red_herring_services:
                    self.red_herring_actions += 1
                    delta = -0.04
                elif target not in self.config.all_services:
                    self.wrong_actions_count += 1
                    delta = -0.02
                else:
                    # Valid service but not a golden action
                    self.wrong_actions_count += 1
                    delta = 0.0

        # ---- update_status ----
        elif tool == "update_status":
            message = args.get("message", args.get("text", args.get("content", "")))
            self.status_updates.append((str(message), step_number))
            # Quality bonus: does the update mention a relevant service?
            if any(svc in str(message) for svc in self.config.relevant_services):
                delta = 0.015
            else:
                delta = 0.005

        # ---- declare_resolved ----
        elif tool == "declare_resolved":
            # Episode-end reward is computed separately via compute_final_reward
            delta = 0.0

        # ---- default / unknown tool ----
        else:
            delta = 0.0

        self.cumulative_step_reward += delta
        return delta

    def on_sla_violation_step(self) -> float:
        """
        Call this each step when the SLA metric is breached.
        Returns a small per-step penalty.
        """
        self.sla_violation_steps += 1
        return -0.004

    def mark_fix_verified(self) -> None:
        """
        Call this when the agent queries metrics AFTER applying a correct action.
        Signals that the agent validated the fix rather than assuming it worked.
        """
        self.fix_verified = True

    # -----------------------------------------------------------------------
    # Episode-end reward computation
    # -----------------------------------------------------------------------

    def compute_final_reward(
        self,
        root_cause: str,
        total_steps: int,
        belief_engine: "Optional[BeliefEngine]" = None,
        workflow_machine: "Optional[WorkflowMachine]" = None,
    ) -> tuple[float, RewardBreakdown]:
        """
        Compute all episode-end rewards including D7 (Epistemic Quality) and D8 (Workflow Coherence).

        Parameters
        ----------
        root_cause : str
            The root-cause string supplied in declare_resolved.
        total_steps : int
            Total number of steps in the episode.
        belief_engine : BeliefEngine, optional
            For D7 computation.
        workflow_machine : WorkflowMachine, optional
            For D8 computation. Also gates fix_verification via skipped_verify flag.

        Returns
        -------
        (final_score, breakdown)
            final_score is in [0.0, 1.0], rounded to 3 decimal places.
        """
        cfg = self.config
        w = cfg.weights
        bd = RewardBreakdown()

        # Store max budgets for feedback display
        bd.components["_d1_max"] = w.get("situational", 0.11)
        bd.components["_d2_max"] = w.get("diagnostic", 0.19)
        bd.components["_d3_max"] = w.get("remediation", 0.25)
        bd.components["_d4_max"] = w.get("time", 0.10)
        bd.components["_d5_max"] = w.get("communication", 0.10)
        bd.components["_d7_max"] = w.get("epistemic_quality", 0.08)
        bd.components["_d8_max"] = w.get("workflow_coherence", 0.07)

        # ----------------------------------------------------------------
        # D1 — Situational Awareness
        # ----------------------------------------------------------------
        d1_max = w.get("situational", 0.11)

        # alert_ack (20% of D1)
        if self.alert_ack_step is not None:
            if self.alert_ack_step <= 1:
                ack_frac = 1.0
            elif self.alert_ack_step <= 3:
                ack_frac = 0.5
            else:
                ack_frac = 0.0
        else:
            ack_frac = 0.0
        alert_ack_score = ack_frac * 0.20 * d1_max

        # blast_radius (40% of D1): fraction of relevant services investigated
        total_relevant = len(cfg.relevant_services)
        investigated_relevant = len(self.relevant_services_investigated)
        blast_frac = investigated_relevant / max(1, total_relevant)
        blast_radius_score = blast_frac * 0.40 * d1_max

        # investigation_depth (40% of D1): relevant queries / total relevant services
        rel_ratio = investigated_relevant / max(1, total_relevant)
        rel_ratio = min(1.0, rel_ratio)
        inv_depth_score = rel_ratio * 0.40 * d1_max

        bd.situational = alert_ack_score + blast_radius_score + inv_depth_score
        bd.components["alert_ack"] = alert_ack_score
        bd.components["blast_radius"] = blast_radius_score
        bd.components["investigation_depth"] = inv_depth_score
        bd.components["_alert_ack_step"] = self.alert_ack_step if self.alert_ack_step is not None else "N/A"
        bd.components["_relevant_investigated"] = investigated_relevant
        bd.components["_relevant_total"] = total_relevant
        bd.components["_rel_ratio"] = rel_ratio

        # ----------------------------------------------------------------
        # D2 — Diagnostic Quality
        # ----------------------------------------------------------------
        d2_max = w.get("diagnostic", 0.19)

        # root_cause_id (40% of D2)
        root_cause_lower = root_cause.lower()
        kw_matches = sum(
            1 for kw in cfg.root_cause_keywords
            if kw.lower() in root_cause_lower
        )
        if kw_matches >= 2:
            rc_frac = 1.0
        elif kw_matches == 1:
            rc_frac = 0.5
        else:
            rc_frac = 0.0
        root_cause_id_score = rc_frac * 0.40 * d2_max

        # investigation_efficiency (20% of D2)
        inv_eff_ratio = min(
            1.0,
            len(self.relevant_services_investigated) / max(1, self.investigation_steps),
        )
        inv_eff_score = inv_eff_ratio * 0.20 * d2_max

        # hypothesis_coherence (20% of D2): was root_cause_service investigated before any remediation?
        rcs_investigated_before_fix = (
            cfg.root_cause_service in self.relevant_services_investigated
            and (
                self._first_relevant_inv_step is not None
                and (
                    self.first_correct_action_step is None
                    or self._first_relevant_inv_step <= self.first_correct_action_step
                )
            )
        )
        hyp_frac = 1.0 if rcs_investigated_before_fix else 0.0
        hypothesis_coherence_score = hyp_frac * 0.20 * d2_max

        # red_herring_resistance (20% of D2)
        total_remediations = len(self.actions_taken)
        rh_resistance = 1.0 - (self.red_herring_actions / max(1, total_remediations)) if total_remediations > 0 else 1.0
        rh_resistance = max(0.0, rh_resistance)
        rh_score = rh_resistance * 0.20 * d2_max

        bd.diagnostic = root_cause_id_score + inv_eff_score + hypothesis_coherence_score + rh_score
        bd.components["root_cause_id"] = root_cause_id_score
        bd.components["investigation_efficiency"] = inv_eff_score
        bd.components["hypothesis_coherence"] = hypothesis_coherence_score
        bd.components["red_herring_resistance"] = rh_score
        bd.components["_kw_matches"] = kw_matches
        bd.components["_inv_eff_ratio"] = inv_eff_ratio
        bd.components["_rh_actions"] = self.red_herring_actions

        # ----------------------------------------------------------------
        # D3 — Remediation Quality
        # ----------------------------------------------------------------
        d3_max = w.get("remediation", 0.25)

        # action_correctness (40% of D3)
        golden_count = len(cfg.golden_actions)
        correct_count = len(self.correct_actions_applied)
        ac_frac = correct_count / max(1, golden_count)
        action_correctness_score = ac_frac * 0.40 * d3_max

        # action_ordering (20% of D3)
        ordering_score: float
        if not cfg.action_order_constraints:
            ordering_score = 1.0 * 0.20 * d3_max
        else:
            satisfied = 0
            for (i, j) in cfg.action_order_constraints:
                pos_i = _find_golden_action_position(cfg.golden_actions[i], self.actions_taken)
                pos_j = _find_golden_action_position(cfg.golden_actions[j], self.actions_taken)
                if pos_i is not None and pos_j is not None and pos_i < pos_j:
                    satisfied += 1
                elif pos_i is None and pos_j is None:
                    pass  # Neither applied; don't penalise
            ord_frac = satisfied / max(1, len(cfg.action_order_constraints))
            ordering_score = ord_frac * 0.20 * d3_max

        # collateral_damage (20% of D3)
        collateral_score = max(0.0, 1.0 - 0.3 * self.wrong_actions_count) * 0.20 * d3_max

        # fix_verification (20% of D3)
        # If workflow machine says the agent skipped verify, zero out fix_verification
        skipped_verify = workflow_machine.skipped_verify if workflow_machine else False
        fv_score = (1.0 if (self.fix_verified and not skipped_verify) else 0.0) * 0.20 * d3_max

        bd.remediation = action_correctness_score + ordering_score + collateral_score + fv_score
        bd.components["action_correctness"] = action_correctness_score
        bd.components["action_ordering"] = ordering_score
        bd.components["collateral_damage"] = collateral_score
        bd.components["fix_verification"] = fv_score
        bd.components["_correct_applied"] = correct_count
        bd.components["_golden_total"] = golden_count
        bd.components["_wrong_actions"] = self.wrong_actions_count
        bd.components["_ord_max"] = ordering_score  # for to_feedback comparison

        # ----------------------------------------------------------------
        # D4 — Time Efficiency
        # ----------------------------------------------------------------
        d4_max = w.get("time", 0.10)

        # mttd (35% of D4): steps to first relevant service investigation
        if self._first_relevant_inv_step is not None:
            investigation_delay = self._first_relevant_inv_step
            mttd_score = max(0.0, 1.0 - investigation_delay / 8.0) * 0.35 * d4_max
        else:
            mttd_score = 0.0
            investigation_delay = None

        # mttr (35% of D4): steps between first relevant investigation and first correct action
        if (
            self._first_relevant_inv_step is not None
            and self.first_correct_action_step is not None
        ):
            remediation_delay = max(0, self.first_correct_action_step - self._first_relevant_inv_step)
            mttr_score = max(0.0, 1.0 - remediation_delay / 10.0) * 0.35 * d4_max
        else:
            remediation_delay = None
            mttr_score = 0.0

        # sla_compliance (30% of D4)
        effective_total = max(1, total_steps)
        capped_violations = min(self.sla_violation_steps, effective_total)
        sla_comp_frac = 1.0 - (capped_violations / effective_total)
        sla_compliance_score = max(0.0, sla_comp_frac) * 0.30 * d4_max

        bd.time_efficiency = mttd_score + mttr_score + sla_compliance_score
        bd.components["mttd"] = mttd_score
        bd.components["mttr"] = mttr_score
        bd.components["sla_compliance"] = sla_compliance_score
        bd.components["_first_rel_step"] = self._first_relevant_inv_step
        bd.components["_rem_delay"] = remediation_delay
        bd.components["_sla_viol_steps"] = self.sla_violation_steps
        bd.components["_total_steps"] = total_steps

        # ----------------------------------------------------------------
        # D5 — Communication
        # ----------------------------------------------------------------
        d5_max = w.get("communication", 0.10)
        n_updates = len(self.status_updates)

        # update_cadence (50% of D5)
        if n_updates == 0:
            cadence_frac = 0.0
        elif n_updates == 1:
            cadence_frac = 0.5
        else:
            cadence_frac = 1.0
        cadence_score = cadence_frac * 0.50 * d5_max

        # update_quality (30% of D5)
        if n_updates > 0:
            quality_keywords = list(cfg.relevant_services) + [cfg.root_cause_type]
            quality_count = sum(
                1 for msg, _ in self.status_updates
                if any(kw.lower() in msg.lower() for kw in quality_keywords)
            )
            quality_frac = min(1.0, quality_count / n_updates)
        else:
            quality_frac = 0.0
        quality_score = quality_frac * 0.30 * d5_max

        # resolution_accuracy (20% of D5)
        rc_words = root_cause.strip().split()
        if len(rc_words) >= 10:
            ra_frac = 1.0
        elif len(rc_words) > 0:
            ra_frac = 0.5
        else:
            ra_frac = 0.0
        ra_score = ra_frac * 0.20 * d5_max

        bd.communication = cadence_score + quality_score + ra_score
        bd.components["update_cadence"] = cadence_score
        bd.components["update_quality"] = quality_score
        bd.components["resolution_accuracy"] = ra_score
        bd.components["_n_updates"] = n_updates

        # ----------------------------------------------------------------
        # D6 — Anti-pattern Penalties
        # ----------------------------------------------------------------
        ap_cap = w.get("anti_patterns", 0.10)

        blind_pen = min(0.06, 0.03 * self.remediations_before_investigation)
        circular_pen = min(0.04, 0.01 * self.circular_query_count)
        rh_action_pen = min(0.08, 0.04 * self.red_herring_actions)
        unresolved_pen = 0.05 if len(self.correct_actions_applied) == 0 else 0.0

        raw_penalty = blind_pen + circular_pen + rh_action_pen + unresolved_pen
        capped_penalty = min(raw_penalty, ap_cap)
        bd.anti_patterns = -capped_penalty

        bd.penalties["blind_remediations"] = -blind_pen
        bd.penalties["circular_queries"] = -circular_pen
        bd.penalties["red_herring_actions"] = -rh_action_pen
        bd.penalties["unresolved_penalty"] = -unresolved_pen

        # ----------------------------------------------------------------
        # D7 — Epistemic Quality
        # ----------------------------------------------------------------
        d7_max = w.get("epistemic_quality", 0.08)

        if belief_engine is not None:
            cum_gain = belief_engine.cumulative_gain()
            n_hyp = max(2, len(belief_engine.hypothesis_space))
            # Normalize: max possible is ~0.04 * n_hyp (one high-gain query per hypothesis)
            gain_norm = min(1.0, cum_gain / (0.04 * n_hyp))
            cum_gain_score = gain_norm * 0.40 * d7_max

            true_conf = belief_engine.final_confidence_on_true()
            true_conf_score = true_conf * 0.40 * d7_max

            redundancy_ratio = belief_engine.redundancy_ratio()
            # Penalty scales from 0 (no redundancy) to -0.20*d7_max (all queries redundant)
            redundancy_score = max(0.0, 1.0 - redundancy_ratio * 2.0) * 0.20 * d7_max

            bd.epistemic_quality = cum_gain_score + true_conf_score + redundancy_score
            bd.components["cumulative_epistemic_gain"] = cum_gain_score
            bd.components["final_belief_confidence"] = true_conf_score
            bd.components["redundancy_penalty"] = redundancy_score
            bd.components["_cum_gain_raw"] = cum_gain
            bd.components["_true_conf"] = true_conf
            bd.components["_redundancy_ratio"] = redundancy_ratio
        else:
            bd.epistemic_quality = 0.0
            bd.components["cumulative_epistemic_gain"] = 0.0
            bd.components["final_belief_confidence"] = 0.0
            bd.components["redundancy_penalty"] = 0.0
            bd.components["_cum_gain_raw"] = 0.0
            bd.components["_true_conf"] = 0.0
            bd.components["_redundancy_ratio"] = 0.0

        # ----------------------------------------------------------------
        # D8 — Workflow Coherence
        # ----------------------------------------------------------------
        d8_max = w.get("workflow_coherence", 0.07)

        if workflow_machine is not None:
            prog = workflow_machine.phase_progression_score()  # 0.0–1.0
            phase_prog_score = prog * 0.50 * d8_max

            verify_bonus = (0.50 * d8_max) if workflow_machine.verify_before_resolve() else 0.0

            bd.workflow_coherence = phase_prog_score + verify_bonus
            bd.components["phase_progression"] = phase_prog_score
            bd.components["verify_bonus"] = verify_bonus
            bd.components["_phase_summary"] = workflow_machine.phases_summary()
        else:
            bd.workflow_coherence = 0.0
            bd.components["phase_progression"] = 0.0
            bd.components["verify_bonus"] = 0.0
            bd.components["_phase_summary"] = "N/A"

        # ----------------------------------------------------------------
        # Narrative hints
        # ----------------------------------------------------------------
        hints: list[str] = []
        score = bd.total
        if score >= 0.90:
            hints.append("Outstanding incident response! All major dimensions handled well.")
        elif score >= 0.75:
            hints.append("Excellent investigation! Post-fix verification would push this score higher.")
        elif score >= 0.60:
            hints.append("Good response. Focus on faster diagnosis and verifying fixes.")
        elif score >= 0.40:
            hints.append("Adequate response, but investigate relevant services more thoroughly.")
        else:
            hints.append("Significant improvement needed: avoid blind remediations and cover more relevant services.")

        if not self.fix_verified:
            hints.append("Tip: query metrics after a remediation to verify the fix worked.")
        if self.remediations_before_investigation > 0:
            hints.append(
                f"Warning: {self.remediations_before_investigation} remediation(s) applied before any investigation."
            )
        if workflow_machine and workflow_machine.skipped_verify:
            hints.append("Tip: reach the VERIFY phase (query metrics after fix) before declaring resolved.")
        if belief_engine and belief_engine.redundancy_ratio() > 0.3:
            hints.append("Tip: avoid querying the same service repeatedly — each query should reveal new information.")
        bd._narrative_hints = hints

        # ----------------------------------------------------------------
        # Final score
        # ----------------------------------------------------------------
        episode_score = (
            bd.situational
            + bd.diagnostic
            + bd.remediation
            + bd.time_efficiency
            + bd.communication
            + bd.anti_patterns
            + bd.epistemic_quality
            + bd.workflow_coherence
        )
        final_score = round(max(0.0, min(1.0, episode_score)), 3)
        return final_score, bd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_service(args: dict) -> str:
    """Extract the service name from a tool args dict (handles common key names)."""
    for key in ("service", "service_name", "target", "source"):
        if key in args:
            return str(args[key])
    # Fallback: return first string value
    for v in args.values():
        if isinstance(v, str):
            return v
    return "unknown"


def _find_golden_action_position(
    golden: tuple[str, str],
    actions_taken: list[tuple[str, str]],
) -> Optional[int]:
    """
    Return the index in actions_taken where golden action was applied (first match),
    or None if not found. Uses fuzzy target matching consistent with on_action().
    """
    g_type, g_target = golden
    for idx, (a_type, a_target) in enumerate(actions_taken):
        if a_target in g_target or g_target in a_target or a_target.lower() == g_target.lower():
            return idx
    return None
