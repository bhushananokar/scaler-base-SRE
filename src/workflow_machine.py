"""
workflow_machine.py
===================
SRE Workflow Coherence Machine — finite state machine tracking the SRE investigation
phase sequence. Phase transitions earn positive rewards; violating the expected ordering
earns penalties. The machine's current state is surfaced in the agent observation.

Phases (in order)
-----------------
OBSERVE     — agent has seen alerts but not yet investigated any service
HYPOTHESIZE — agent has done at least one relevant log/metric/runbook query
DIAGNOSE    — agent has investigated the root_cause_service specifically
REMEDIATE   — agent has applied at least one remediation action
VERIFY      — agent has queried metrics/logs after remediation to confirm recovery

Transition rewards
------------------
OBSERVE     → HYPOTHESIZE : +0.02  (first investigation query)
HYPOTHESIZE → DIAGNOSE    : +0.03  (root cause service specifically queried)
DIAGNOSE    → REMEDIATE   : +0.03  (correct remediation applied after diagnosis)
REMEDIATE   → VERIFY      : +0.02  (metrics/logs queried after remediation)

Penalties
---------
execute_remediation from OBSERVE (blind) : -0.05
declare_resolved from OBSERVE/HYPOTHESIZE/DIAGNOSE : -0.04 + sets skipped_verify flag
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

class Phase:
    OBSERVE     = "OBSERVE"
    HYPOTHESIZE = "HYPOTHESIZE"
    DIAGNOSE    = "DIAGNOSE"
    REMEDIATE   = "REMEDIATE"
    VERIFY      = "VERIFY"

    _ORDER = [OBSERVE, HYPOTHESIZE, DIAGNOSE, REMEDIATE, VERIFY]

    @classmethod
    def index(cls, phase: str) -> int:
        try:
            return cls._ORDER.index(phase)
        except ValueError:
            return -1


# ---------------------------------------------------------------------------
# WorkflowMachine
# ---------------------------------------------------------------------------

class WorkflowMachine:
    """
    Finite state machine tracking SRE investigation phase adherence.

    Parameters
    ----------
    root_cause_service : str
        The service that is the root cause of the incident (e.g. "payment-service").
        Used to detect HYPOTHESIZE → DIAGNOSE transitions.
    """

    def __init__(self, root_cause_service: str) -> None:
        self.root_cause_service = root_cause_service.lower().strip()
        self.phase: str = Phase.OBSERVE

        # Ordered list of phases reached (for scoring)
        self._phases_reached: list[str] = [Phase.OBSERVE]

        # Flags used in episode-end scoring
        self.skipped_verify: bool = False       # declared resolved without reaching VERIFY
        self.premature_blind: bool = False      # remediated from OBSERVE
        self.verification_detour: bool = False  # went REMEDIATE → HYPOTHESIZE/DIAGNOSE without verifying

        # Internal tracking
        self._correct_action_ever_applied: bool = False
        self._in_remediate_then_back: bool = False  # re-entered DIAGNOSE from REMEDIATE

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def on_action(self, tool: str, args: dict, is_correct_action: bool = False) -> float:
        """
        Process one agent action.

        Parameters
        ----------
        tool : str
            The tool name (e.g. "query_logs", "execute_remediation").
        args : dict
            Tool arguments (used to check which service was queried).
        is_correct_action : bool
            True if execute_remediation matched a golden action. Supplied by BaseIncident.

        Returns
        -------
        float
            Per-step reward delta (positive for valid transitions, negative for violations).
        """
        reward = 0.0

        if tool in ("query_logs", "query_metrics"):
            reward = self._handle_query(args)

        elif tool in ("read_runbook", "check_deploy_history"):
            reward = self._handle_investigation(tool, args)

        elif tool == "execute_remediation":
            reward = self._handle_remediation(is_correct_action)

        elif tool == "declare_resolved":
            reward = self._handle_declare_resolved()

        return reward

    # ------------------------------------------------------------------
    # Per-tool handlers
    # ------------------------------------------------------------------

    def _handle_query(self, args: dict) -> float:
        """Handle query_logs / query_metrics."""
        service = args.get("service", "").lower().strip()
        reward = 0.0

        if self.phase == Phase.OBSERVE:
            self._transition(Phase.HYPOTHESIZE)
            reward = 0.02

        elif self.phase == Phase.HYPOTHESIZE:
            # Transition to DIAGNOSE when root cause service is specifically investigated
            if self.root_cause_service and self.root_cause_service in service:
                self._transition(Phase.DIAGNOSE)
                reward = 0.03

        elif self.phase == Phase.REMEDIATE:
            if self._correct_action_ever_applied:
                # Valid: verifying the fix
                self._transition(Phase.VERIFY)
                reward = 0.02
            else:
                # Agent remediates something wrong and goes back to investigate
                self.verification_detour = True
                # Re-enter DIAGNOSE to allow continued investigation
                self.phase = Phase.DIAGNOSE
                if Phase.DIAGNOSE not in self._phases_reached:
                    self._phases_reached.append(Phase.DIAGNOSE)

        # From DIAGNOSE or VERIFY: additional queries are free (no reward, no penalty)
        return reward

    def _handle_investigation(self, tool: str, args: dict) -> float:
        """Handle read_runbook / check_deploy_history."""
        service = args.get("service", args.get("topic", "")).lower().strip()
        reward = 0.0

        if self.phase == Phase.OBSERVE:
            self._transition(Phase.HYPOTHESIZE)
            reward = 0.02

        elif self.phase == Phase.HYPOTHESIZE:
            # check_deploy_history on root cause service triggers DIAGNOSE
            if tool == "check_deploy_history" and self.root_cause_service and self.root_cause_service in service:
                self._transition(Phase.DIAGNOSE)
                reward = 0.03

        return reward

    def _handle_remediation(self, is_correct_action: bool) -> float:
        """Handle execute_remediation."""
        reward = 0.0

        if self.phase == Phase.OBSERVE:
            # Blind remediation — no investigation at all
            self.premature_blind = True
            reward = -0.05

        elif self.phase in (Phase.HYPOTHESIZE, Phase.DIAGNOSE):
            if is_correct_action:
                self._correct_action_ever_applied = True
                self._transition(Phase.REMEDIATE)
                reward = 0.03
            # Wrong remediation: stay in current phase, no transition reward

        elif self.phase == Phase.REMEDIATE:
            if is_correct_action:
                # Additional correct remediation (e.g. cascade: restart + scale)
                self._correct_action_ever_applied = True
                reward = 0.01  # small bonus for multi-step correct remediation

        elif self.phase == Phase.VERIFY:
            # Remediating again after verify (valid in cascade scenarios)
            if is_correct_action:
                self._correct_action_ever_applied = True
                self.phase = Phase.REMEDIATE  # step back to REMEDIATE
                reward = 0.01

        return reward

    def _handle_declare_resolved(self) -> float:
        """Handle declare_resolved."""
        if self.phase in (Phase.OBSERVE, Phase.HYPOTHESIZE, Phase.DIAGNOSE):
            self.skipped_verify = True
            return -0.04
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_phase: str) -> None:
        self.phase = new_phase
        if new_phase not in self._phases_reached:
            self._phases_reached.append(new_phase)

    # ------------------------------------------------------------------
    # Episode-end scoring helpers
    # ------------------------------------------------------------------

    def phase_progression_score(self) -> float:
        """
        Fraction of the canonical five phases reached in order.
        0.2 per phase reached, max 1.0.
        """
        count = sum(1 for p in Phase._ORDER if p in self._phases_reached)
        return count * 0.2

    def skipped_phases_count(self) -> int:
        """Number of canonical phases NOT reached during the episode."""
        return sum(1 for p in Phase._ORDER if p not in self._phases_reached)

    def verify_before_resolve(self) -> bool:
        """True if the agent passed through VERIFY before declaring resolved."""
        return Phase.VERIFY in self._phases_reached and not self.skipped_verify

    def phases_summary(self) -> str:
        """Human-readable phase progression string, e.g. 'OBSERVE→HYPOTHESIZE→DIAGNOSE→REMEDIATE'."""
        return "→".join(self._phases_reached)
