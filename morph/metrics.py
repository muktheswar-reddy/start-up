"""Behaviour comparator.

Measures how model behaviour changes across composition configurations
using task-specific metrics and distribution distances.  Every metric is
computed with its uncertainty estimate so findings can be reported with
confidence intervals rather than point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class MetricResult:
    """One measurement with its uncertainty."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "confidence": float(self.confidence),
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


def _bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1.0 - confidence) / 2.0
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def label_agreement_rate(observed: list[Any], reference: list[Any]) -> MetricResult:
    """Fraction of probes where the composed model agrees with the reference."""
    observed = list(observed)
    reference = list(reference)
    if not observed:
        raise ValueError("no observations provided")
    matches = [1.0 if o == r else 0.0 for o, r in zip(observed, reference)]
    arr = np.array(matches)
    lo, hi = _bootstrap_ci(arr)
    return MetricResult(
        name="label_agreement_rate",
        value=float(arr.mean()),
        ci_low=lo,
        ci_high=hi,
        unit="fraction",
        metadata={"n": len(observed)},
    )


def label_flip_rate(observed: list[Any], reference: list[Any]) -> MetricResult:
    """Fraction of probes where the composed model changed its label."""
    observed = list(observed)
    reference = list(reference)
    if not observed:
        raise ValueError("no observations provided")
    flips = [1.0 if o != r else 0.0 for o, r in zip(observed, reference)]
    arr = np.array(flips)
    lo, hi = _bootstrap_ci(arr)
    return MetricResult(
        name="label_flip_rate",
        value=float(arr.mean()),
        ci_low=lo,
        ci_high=hi,
        unit="fraction",
        metadata={"n": len(observed)},
    )


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence between two normalised discrete distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (bounded, symmetric)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def distribution_distance(
    observed: list[np.ndarray],
    reference: list[np.ndarray],
    *,
    distance: str = "js",
    eps: float = 1e-12,
) -> MetricResult:
    """Mean distribution distance between composed and reference outputs."""
    if len(observed) != len(reference):
        raise ValueError("observed and reference must have equal length")
    if not observed:
        raise ValueError("no observations provided")

    fn = {"js": js_divergence, "kl": kl_divergence}.get(distance)
    if fn is None:
        raise ValueError(f"unsupported distance: {distance!r}")

    values = np.array([fn(o, r) for o, r in zip(observed, reference)])
    lo, hi = _bootstrap_ci(values)
    return MetricResult(
        name=f"{distance}_divergence",
        value=float(values.mean()),
        ci_low=lo,
        ci_high=hi,
        unit="bits" if distance == "js" else "nats",
        metadata={"n": len(observed), "distance": distance},
    )


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """T-distribution confidence interval for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (float(arr[0]) if arr.size else float("nan"),) * 2  # type: ignore[return-value]
    mean = float(arr.mean())
    sem = float(stats.sem(arr))
    if sem == 0:
        return mean, mean
    tcrit = float(stats.t.ppf(1 - (1 - confidence) / 2, df=arr.size - 1))
    return mean - tcrit * sem, mean + tcrit * sem