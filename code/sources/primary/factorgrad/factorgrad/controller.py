"""Deterministic Monte Carlo schedules for FactorGrad-GS."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


_UINT64_MASK = (1 << 64) - 1
_TIME_SALT = 0xD1B54A32D192ED03
_CHANNEL_SALT = 0xA24BAED4963EE407
_STRIDE = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    value = (value + _STRIDE) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _probability_ratio(probability: float, max_denominator: int = 64) -> Tuple[int, int]:
    probability = float(probability)
    if not 0.0 < probability <= 1.0:
        raise ValueError("frozen_probability must be in (0, 1]")
    ratio = Fraction(probability).limit_denominator(max_denominator)
    if not math.isclose(probability, float(ratio), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "frozen_probability must be representable with denominator <= {}".format(
                max_denominator
            )
        )
    return ratio.numerator, ratio.denominator


def _permutation(size: int, key: int) -> List[int]:
    """Generate a local deterministic permutation without touching global RNG state."""
    ranked = []
    for index in range(size):
        sample = _splitmix64((key + index * _STRIDE) & _UINT64_MASK)
        ranked.append((sample, index))
    ranked.sort()
    return [index for _, index in ranked]


@dataclass(frozen=True)
class FactorDecision:
    iteration: int
    phase: str
    mode: str
    build_structural_graph: bool
    selected_channel: Optional[int]
    temporal_probability: float
    temporal_weight: float


class FactorGradController:
    """Densification-aware time x RGB structural-gradient scheduler.

    ``late`` preserves the conservative ablation: full RGB gradients are used until
    densification and stabilization finish. ``full`` additionally factorizes RGB
    gradients from ``early_start_iteration`` until densification ends, while retaining
    every temporal update in that topology-active phase. Both schedules use the same
    stratified time/channel estimator after stabilization.
    """

    def __init__(
        self,
        enabled: bool = False,
        topology_end_iteration: int = 15_000,
        stabilization_steps: int = 768,
        frozen_probability: float = 2.0 / 3.0,
        seed: int = 3407,
        schedule: str = "late",
        early_start_iteration: int = 3_000,
        channel_sampling: bool = True,
        temporal_sampling: bool = True,
        topology_protection: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.topology_end_iteration = int(topology_end_iteration)
        self.stabilization_steps = int(stabilization_steps)
        self.frozen_probability = float(frozen_probability)
        self.seed = int(seed)
        self.schedule = str(schedule)
        self.early_start_iteration = int(early_start_iteration)
        self.channel_sampling = bool(channel_sampling)
        self.temporal_sampling = bool(temporal_sampling)
        self.topology_protection = bool(topology_protection)
        if self.topology_end_iteration < 0:
            raise ValueError("topology_end_iteration must be non-negative")
        if self.stabilization_steps < 0:
            raise ValueError("stabilization_steps must be non-negative")
        if self.schedule not in ("late", "full"):
            raise ValueError("schedule must be 'late' or 'full'")
        if self.early_start_iteration < 0:
            raise ValueError("early_start_iteration must be non-negative")
        if (
            self.schedule == "full"
            and self.early_start_iteration > self.topology_end_iteration
        ):
            raise ValueError(
                "early_start_iteration must be between 0 and topology_end_iteration"
            )
        self._time_numerator, self._time_denominator = _probability_ratio(
            self.frozen_probability
        )
        self.freeze_iteration = self.topology_end_iteration + self.stabilization_steps
        self.mode_counts = Counter()
        self.phase_counts = Counter()
        self.phase_mode_counts = Counter()
        self.outcome_counts = Counter()
        self.channel_counts = Counter()
        self.selected_applied_updates = 0

    def _time_selected(self, iteration: int, start_iteration: int) -> bool:
        if not self.temporal_sampling or self._time_numerator == self._time_denominator:
            return True
        active_index = iteration - start_iteration
        block_index = active_index // self._time_denominator
        position = active_index % self._time_denominator
        key = (
            self.seed + _TIME_SALT + block_index * _STRIDE
        ) & _UINT64_MASK
        selected_positions = set(
            _permutation(self._time_denominator, key)[: self._time_numerator]
        )
        return position in selected_positions

    def _next_channel(self) -> int:
        block_index = self.selected_applied_updates // 3
        position = self.selected_applied_updates % 3
        key = (
            self.seed + _CHANNEL_SALT + block_index * _STRIDE
        ) & _UINT64_MASK
        return _permutation(3, key)[position]

    def _selected_structural_decision(
        self,
        iteration: int,
        phase: str,
        temporal_probability: float,
        temporal_weight: float,
    ) -> FactorDecision:
        if self.channel_sampling:
            return FactorDecision(
                iteration=iteration,
                phase=phase,
                mode="factorized_channel",
                build_structural_graph=True,
                selected_channel=self._next_channel(),
                temporal_probability=temporal_probability,
                temporal_weight=temporal_weight,
            )
        return FactorDecision(
            iteration=iteration,
            phase=phase,
            mode=(
                "reweighted_full_rgb"
                if not math.isclose(temporal_weight, 1.0)
                else "full_rgb"
            ),
            build_structural_graph=True,
            selected_channel=None,
            temporal_probability=temporal_probability,
            temporal_weight=temporal_weight,
        )

    def decision(self, iteration: int) -> FactorDecision:
        iteration = int(iteration)
        if not self.enabled:
            return FactorDecision(
                iteration=iteration,
                phase="disabled",
                mode="baseline_full",
                build_structural_graph=True,
                selected_channel=None,
                temporal_probability=1.0,
                temporal_weight=1.0,
            )
        if not self.topology_protection:
            if iteration < self.early_start_iteration:
                return FactorDecision(
                    iteration=iteration,
                    phase="initial_warmup",
                    mode="full_rgb",
                    build_structural_graph=True,
                    selected_channel=None,
                    temporal_probability=1.0,
                    temporal_weight=1.0,
                )
            probability = self.frozen_probability if self.temporal_sampling else 1.0
            if self._time_selected(iteration, self.early_start_iteration):
                return self._selected_structural_decision(
                    iteration=iteration,
                    phase="unprotected_budgeted",
                    temporal_probability=probability,
                    temporal_weight=1.0 / probability,
                )
            return FactorDecision(
                iteration=iteration,
                phase="unprotected_budgeted",
                mode="l1_only",
                build_structural_graph=False,
                selected_channel=None,
                temporal_probability=probability,
                temporal_weight=0.0,
            )
        if (
            self.schedule == "full"
            and self.early_start_iteration <= iteration < self.topology_end_iteration
        ):
            return self._selected_structural_decision(
                iteration=iteration,
                phase="topology_active",
                temporal_probability=1.0,
                temporal_weight=1.0,
            )
        if iteration < self.freeze_iteration:
            return FactorDecision(
                iteration=iteration,
                phase=(
                    "topology_warmup"
                    if iteration < self.topology_end_iteration
                    else "post_topology_stabilization"
                ),
                mode="full_rgb",
                build_structural_graph=True,
                selected_channel=None,
                temporal_probability=1.0,
                temporal_weight=1.0,
            )
        probability = self.frozen_probability if self.temporal_sampling else 1.0
        if self._time_selected(iteration, self.freeze_iteration):
            return self._selected_structural_decision(
                iteration=iteration,
                phase="topology_frozen",
                temporal_probability=probability,
                temporal_weight=1.0 / probability,
            )
        return FactorDecision(
            iteration=iteration,
            phase="topology_frozen",
            mode="l1_only",
            build_structural_graph=False,
            selected_channel=None,
            temporal_probability=probability,
            temporal_weight=0.0,
        )

    def record(self, decision: FactorDecision, did_backward: bool) -> None:
        self.mode_counts[decision.mode] += 1
        self.phase_counts[decision.phase] += 1
        self.phase_mode_counts[
            "{}:{}".format(decision.phase, decision.mode)
        ] += 1
        self.outcome_counts["backward" if did_backward else "backward_skipped"] += 1
        if decision.mode == "factorized_channel":
            channel = int(decision.selected_channel)
            status = "applied" if did_backward else "discarded"
            self.channel_counts["{}:{}".format(status, channel)] += 1
            if did_backward:
                self.selected_applied_updates += 1
        elif decision.mode == "l1_only" and did_backward:
            self.outcome_counts["l1_only_backward"] += 1

    def state_dict(self) -> Dict[str, object]:
        return {
            "selected_applied_updates": self.selected_applied_updates,
            "mode_counts": dict(self.mode_counts),
            "phase_counts": dict(self.phase_counts),
            "phase_mode_counts": dict(self.phase_mode_counts),
            "outcome_counts": dict(self.outcome_counts),
            "channel_counts": dict(self.channel_counts),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.selected_applied_updates = int(state["selected_applied_updates"])
        self.mode_counts = Counter(state.get("mode_counts", {}))
        self.phase_counts = Counter(state.get("phase_counts", {}))
        self.phase_mode_counts = Counter(state.get("phase_mode_counts", {}))
        self.outcome_counts = Counter(state.get("outcome_counts", {}))
        self.channel_counts = Counter(state.get("channel_counts", {}))

    def summary(self) -> Dict[str, object]:
        budget_phases = ("topology_frozen", "unprotected_budgeted")
        frozen_total = sum(
            count
            for key, count in self.phase_mode_counts.items()
            if key.split(":", 1)[0] in budget_phases
        )
        frozen_l1 = sum(
            self.phase_mode_counts["{}:l1_only".format(phase)]
            for phase in budget_phases
        )
        frozen_selected = frozen_total - frozen_l1
        realized_probability = (
            frozen_selected / frozen_total
            if frozen_total
            else None
        )
        return {
            "method": "FactorGrad-GS",
            "enabled": self.enabled,
            "topology_end_iteration": self.topology_end_iteration,
            "stabilization_steps": self.stabilization_steps,
            "freeze_iteration": self.freeze_iteration,
            "frozen_probability": self.frozen_probability,
            "seed": self.seed,
            "schedule": self.schedule,
            "early_start_iteration": self.early_start_iteration,
            "channel_sampling": self.channel_sampling,
            "temporal_sampling": self.temporal_sampling,
            "topology_protection": self.topology_protection,
            "time_ratio": [self._time_numerator, self._time_denominator],
            "selected_applied_updates": self.selected_applied_updates,
            "realized_frozen_probability": realized_probability,
            "mode_counts": dict(self.mode_counts),
            "phase_counts": dict(self.phase_counts),
            "phase_mode_counts": dict(self.phase_mode_counts),
            "outcome_counts": dict(self.outcome_counts),
            "channel_counts": dict(self.channel_counts),
        }

    def save(self, output_directory: str) -> str:
        os.makedirs(output_directory, exist_ok=True)
        path = os.path.join(output_directory, "factorgrad_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
        return path
