"""Interaction & attribution engine.

Estimates non-additive effects; runs controlled ablations/substitutions to
attribute behaviour to components or interactions.

The core attribution strategies are:

1. **Leave-one-component-out (LOCO)** : rebuild the composition without
   component i and measure how the finding changes.

2. **Coefficient perturbation** : move coefficients slightly and measure
   whether the finding is stable or path-dependent.

3. **Matched control** : replace a suspected component with a capability-
   matched control model.

These analyses require the *behaviour comparator* results for each
composition configuration, not the raw model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import MetricResult, confidence_interval


@dataclass
class AttributionFinding:
    """A single attributed finding from an ablation study."""

    composition_index: int
    component_index: int | None  # None if global (not component-specific)
    metric: str
    original_value: float
    modified_value: float | None
    attribution_score: float  # in [-1, 1], negative = component removal *increased* the behaviour
    uncertainty: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "composition_index": self.composition_index,
            "component_index": self.component_index,
            "metric": self.metric,
            "original_value": float(self.original_value),
            "modified_value": float(self.modified_value) if self.modified_value is not None else None,
            "attribution_score": float(self.attribution_score),
            "uncertainty": {"low": self.uncertainty[0], "high": self.uncertainty[1]},
            "metadata": dict(self.metadata),
        }


def leave_one_component_out(
    metric_results: dict[int, list[MetricResult]],
    n_components: int,
    *,
    metric_name: str = "label_flip_rate",
) -> list[AttributionFinding]:
    """LOCO attribution: remove each component in turn and re-measure.

    `metric_results` maps composition index -> list of MetricResults for
    that composition.  For each composition we need its LOCO variants;
    this function produces the attribution scores assuming those variants
    have been run.

    In practice you call this after running an experiment where for each
    composition you also ran the "without component X" variants; the
    `modified_value` field is filled from those runs.
    """
    findings: list[AttributionFinding] = []
    for comp_idx, results in metric_results.items():
        base = next((m for m in results if m.name == metric_name), None)
        if base is None:
            continue
        for comp_i in range(n_components):
            # Placeholder: modified_value would come from the LOCO run.
            findings.append(
                AttributionFinding(
                    composition_index=comp_idx,
                    component_index=comp_i,
                    metric=metric_name,
                    original_value=base.value,
                    modified_value=None,  # Will be filled by caller
                    attribution_score=0.0,  # Will be calculated from LOCO delta
                    uncertainty=(base.ci_low, base.ci_high),
                    metadata={"type": "leave_one_out"},
                )
            )
    return findings


def coefficient_perturbation(
    original: MetricResult,
    perturbed: MetricResult,
    coefficient_delta: float = 0.05,
) -> AttributionFinding:
    """Attribute the metric change to coefficient shift.

    A positive attribution_score means small perturbation *increased* the
    risk metric (suggesting a configuration-sensitive region); negative
    means the region is stable.
    """
    delta = perturbed.value - original.value
    attr = np.clip(delta, -1.0, 1.0)
    return AttributionFinding(
        composition_index=original.metadata.get("composition_index", 0),
        component_index=None,
        metric=original.name,
        original_value=original.value,
        modified_value=perturbed.value,
        attribution_score=float(attr),
        uncertainty=(perturbed.ci_low, perturbed.ci_high),
        metadata={
            "type": "coefficient_perturbation",
            "delta": float(coefficient_delta),
        },
    )


def shapley_decomposition(
    parent_values: dict[str, float],
    composed_value: float,
    metric_name: str,
) -> dict[str, float]:
    """Estimate each component's Shapley value contribution.

    This is a simplified approximation: for n components, compute the
    marginal contribution of adding each component to the empty model.
    For real Shapley, you need all coalitions; this is tractable only
    for very small n.
    """
    n = len(parent_values)
    scores: dict[str, float] = {}
    for name, val in parent_values.items():
        # Marginal contribution: (val - 0) / n, simplified.
        scores[name] = (val - 0) / n
    # Normalize to sum to composed_value.
    total = sum(scores.values())
    if total != 0:
        for k in scores:
            scores[k] *= composed_value / total
    return scores


def estimate_nonadditive_effect(
    child: MetricResult,
    parents: list[MetricResult],
) -> float:
    """Estimate non-additive interaction strength.

    Simple baseline: sum of parent values vs child value.  A positive
    result means the composition is *more* risky than the sum of parts
    (potential synergy or emergence).

    This is a heuristic; proper statistical testing requires a
    controlled experiment with ground-truth labels (see the plan's H1).
    """
    parent_sum = sum(p.value for p in parents if p.value is not None)
    return child.value - parent_sum