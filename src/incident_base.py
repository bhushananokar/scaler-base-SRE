"""
Enhanced BaseIncident — integrates RewardEngine + MetricEngine + BeliefEngine + WorkflowMachine.

Every incident subclass gets:
  - Live metrics that evolve each step (time pressure)
  - Per-step rewards from the 8-dimensional reward engine
  - Epistemic Bayesian reward signal (entropy reduction per investigation)
  - SRE Workflow Coherence Machine rewards (phase transition bonuses/penalties)
  - SLA clock ticking during metric violations
  - Fix-verification detection
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .reward_engine import RewardEngine, IncidentConfig, RewardBreakdown
from .metric_engine import MetricEngine
from .belief_engine import BeliefEngine
from .workflow_machine import WorkflowMachine


@dataclass
class ActionRecord:
    tool: str
    args: dict
    result: str
    step: int
    step_reward: float = 0.0  # per-step reward delta this action earned


class BaseIncident(ABC):
    """
    Base class for incident scenarios.

    Subclasses must:
      1. Call super().__init__(reward_engine, metric_engine, config, belief_engine, workflow_machine)
      2. Implement all abstract methods
      3. Call self._record(tool, args, result, step_reward) to log actions
      4. Call self._after_fix(action, target) when a remediation is applied
    """

    def __init__(
        self,
        reward_engine: RewardEngine,
        metric_engine: MetricEngine,
        config: IncidentConfig,
        belief_engine: BeliefEngine,
        workflow_machine: WorkflowMachine,
        seed: int = 42,
    ):
        self.reward_engine = reward_engine
        self.metric_engine = metric_engine
        self.config = config
        self.belief_engine = belief_engine
        self.workflow_machine = workflow_machine
        self.seed = seed

        self.step_count: int = 0
        self.done: bool = False
        self.action_history: list[ActionRecord] = []
        self._first_correct_action_applied: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self, tool: str, args: dict, step_number: int) -> float:
        """
        Called at the START of every tool dispatch.
        Advances the metric engine, checks SLA, runs epistemic + workflow updates.
        Returns total per-step reward delta.
        """
        # 1. Reward engine: score this action (D1-D6 per-step signals)
        step_reward = self.reward_engine.on_action(tool, args, step_number)

        # 2. Belief engine: Bayesian update for investigation tools
        if tool in ("query_logs", "query_metrics", "read_runbook", "check_deploy_history"):
            epistemic_delta = self.belief_engine.update(tool, args)
            step_reward += epistemic_delta
        elif tool == "execute_remediation":
            # Notify belief engine to track premature remediations
            self.belief_engine.notify_remediation()

        # 3. Workflow machine: phase transition rewards/penalties
        is_correct = getattr(self.reward_engine, '_last_was_correct_action', False)
        workflow_delta = self.workflow_machine.on_action(tool, args, is_correct_action=is_correct)
        step_reward += workflow_delta

        # 4. Metric engine: evolve metrics one step
        self.metric_engine.tick()

        # 5. SLA check: add penalty if currently violated.
        #    Skip terminal step so sla_violation_steps never exceeds total_steps.
        if tool != "declare_resolved" and not self.metric_engine.is_sla_ok():
            step_reward += self.reward_engine.on_sla_violation_step()

        return step_reward

    def _record(self, tool: str, args: dict, result: str, step_reward: float = 0.0):
        self.action_history.append(
            ActionRecord(tool=tool, args=args, result=result,
                        step=self.step_count, step_reward=step_reward)
        )
        self.step_count += 1

    def _after_fix(self, action: str, target: str):
        """Call after applying a remediation. Registers fix with metric engine."""
        self.metric_engine.apply_fix(action, target)
        if not self._first_correct_action_applied:
            self._first_correct_action_applied = True

    def _check_fix_verification(self, tool: str, args: dict):
        """If this is a metrics query after a correct action, mark fix as verified."""
        if (
            tool in ("query_metrics", "query_logs") and
            self._first_correct_action_applied and
            not self.reward_engine.fix_verified
        ):
            self.reward_engine.mark_fix_verified()

    def _actions_of_type(self, tool: str) -> list[ActionRecord]:
        return [a for a in self.action_history if a.tool == tool]

    def _any_action_with(self, tool: str, **kwargs) -> bool:
        for a in self.action_history:
            if a.tool == tool and all(
                str(v).lower() in str(a.args.get(k, "")).lower()
                for k, v in kwargs.items()
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Shared update_status (all tasks use same implementation)
    # ------------------------------------------------------------------

    def update_status(self, message: str) -> tuple[str, float]:
        """Post status update to incident channel. Returns (result, step_reward)."""
        args = {"message": message}
        step_reward = self._tick("update_status", args, self.step_count)
        result = f"[Incident Channel] Status posted: {message}"
        self._record("update_status", args, result, step_reward)
        return result, step_reward

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_task_id(self) -> int: ...

    @abstractmethod
    def get_initial_context(self) -> str: ...

    @abstractmethod
    def list_alerts(self) -> tuple[str, float]: ...

    @abstractmethod
    def query_logs(self, service: str, severity: str, keyword: str) -> tuple[str, float]: ...

    @abstractmethod
    def query_metrics(self, service: str, metric: str) -> tuple[str, float]: ...

    @abstractmethod
    def read_runbook(self, topic: str) -> tuple[str, float]: ...

    @abstractmethod
    def check_deploy_history(self, service: str) -> tuple[str, float]: ...

    @abstractmethod
    def execute_remediation(self, action: str, target: str) -> tuple[str, float]: ...

    @abstractmethod
    def declare_resolved(self, root_cause: str) -> tuple[float, str]: ...
