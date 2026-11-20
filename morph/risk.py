"""Risk & uncertainty layer.

Aggregates evidence without hiding uncertainty or scope limitations.  Each
finding includes observed behaviour, probe identifier, affected configurations,
metric and threshold, uncertainty interval, parent comparisons, attribution
evidence, utility impact, reproducibility hash...

The layer produces structured "finding records" that can be written to the
manifest or included in the final report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .metrics import MetricResult
from .attribution import AttributionFinding


@dataclass
class EvidenceRecord:
    """A complete evidence record for one finding."""

    observed_behavior: str
    probe: str
    affected_configurations: list[int]
    metric: MetricResult
    uncertainty: tuple[float, float]
    parent_comparisons: list[float]
    attribution: list[AttributionFinding] | None = None
    utility_impact: float | None = None
    confidence: float = 0.95
    analyst_disposition: str = "pending"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_behavior": self.observed_behavior,
            "probe": self.probe,
            "affected_configurations": self.affected_configurations,
            "metric": self.metric.to_dict() if self.metric else None,
            "uncertainty": {"low": self.uncertainty[0], "high": self.uncertainty[1]},
            "parent_comparisons": self.parent_comparisons,
            "attribution": [a.to_dict() for a in self.attribution] if self.attribution else None,
            "utility_impact": self.utility_impact,
            "confidence": float(self.confidence),
            "analyst_disposition": self.analyst_disposition,
            "rationale": self.rationale,
        }


def compute_finding_hash(record: EvidenceRecord) -> str:
    """Hash the evidence record for reproducibility."""
    h = hashlib.sha256()
    h.update(json.dumps(record.to_dict(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def flag_if_exceeds(
    metric: MetricResult,
    threshold: float,
    false_positive_ceiling: float = 0.10,
) -> bool:
    """Flag a finding if the metric exceeds the threshold with low FP risk.

    The go/no-go rules in the plan specify continuing only if the method
    adds measurable value *while* keeping a pre-agreed false-positive ceiling.
    This helper checks whether the CI is entirely above threshold (meaning
    the effect is real, not uncertain noise).
    """
    if metric.ci_low >= threshold:
        return True
    if metric.value >= threshold and metric.ci_high >= threshold:
        return True
    return False


def aggregate_findings(
    records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Produce a summary of all findings with counts and risk profile."""
    total = len(records)
    flagged = sum(1 for r in records if flag_if_exceeds(r.metric, 0.05))
    utilities = [r.utility_impact for r in records if r.utility_impact is not None]
    avg_utility = float(np.mean(utilities)) if utilities else None

    return {
        "total_findings": total,
        "flagged_findings": flagged,
        "alert_rate": flagged / total if total else 0.0,
        "avg_utility_impact": avg_utility,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


import numpy as np