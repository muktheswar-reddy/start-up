"""Evidence reporter.

Produces machine-readable JSON/CSV and human-readable JSON/Human-readable PDF/HTML.
The output is called an *evidence bundle* because it contains everything needed
for reproducibility: model hashes, merge recipes, test prompts, versions, metrics,
and uncertainty.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import MergeRecipe, ProbeSpec
from .registry import ModelRegistry, ModelRecord
from .path_sampler import SampledPath
from .metrics import MetricResult
from .attribution import AttributionFinding
from .risk import EvidenceRecord, compute_finding_hash


class EvidenceBundle:
    """A complete evidence bundle that can be exported and reproduced elsewhere."""

    def __init__(
        self,
        registry: ModelRegistry,
        recipe: MergeRecipe,
        paths: list[SampledPath],
        base_metrics: list[MetricResult],
        findings: list[EvidenceRecord],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self.recipe = recipe
        self.paths = paths
        self.base_metrics = base_metrics
        self.findings = findings
        self.config = config or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": "morph-cg-0.1",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config": self.config,
            "registry_manifest": self.registry.manifest(),
            "merge_recipe": self.recipe.to_dict(),
            "sampled_paths": [p.to_dict() for p in self.paths],
            "base_metrics": [m.to_dict() for m in self.base_metrics],
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "total_paths": len(self.paths),
                "total_findings": len(self.findings),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        """Write the bundle to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p

    def to_csv(self) -> list[dict[str, Any]]:
        """Flatten findings to CSV rows."""
        rows = []
        for f in self.findings:
            row = {
                "observed_behavior": f.observed_behavior,
                "probe": f.probe,
                "metric_name": f.metric.name,
                "metric_value": f.metric.value,
                "metric_ci_low": f.metric.ci_low,
                "metric_ci_high": f.metric.ci_high,
                "utility_impact": f.utility_impact,
                "finding_hash": compute_finding_hash(f),
            }
            rows.append(row)
        return rows

    def save_csv(self, path: str | Path) -> Path:
        """Write the flattened findings to CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_csv()
        if rows:
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        return p


def generate_research_memo(bundle: EvidenceBundle) -> str:
    """Generate a plain-text research memo from the evidence bundle."""
    lines = [
        "MORPH-CG Research Memo",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "=" * 60,
        "",
        "## Summary",
        f"- Total composition paths tested: {len(bundle.paths)}",
        f"- Total findings: {len(bundle.findings)}",
        "",
        "## Model Registry",
    ]
    for m in bundle.registry.all():
        lines.append(f"- {m.model_id}: {m.architecture} ({m.file_hash[:8]}...)")
    lines.append("")
    lines.append("## Merge Recipe")
    lines.append(f"Method: {bundle.recipe.method}")
    lines.append(f"Parents: {', '.join(bundle.recipe.parents)}")
    lines.append(f"Coefficients: {bundle.recipe.coefficients}")
    lines.append("")
    lines.append("## Findings")
    for i, f in enumerate(bundle.findings):
        lines.append(f"### Finding {i + 1}")
        lines.append(f"- Behaviour: {f.observed_behavior}")
        lines.append(f"- Probe: {f.probe}")
        lines.append(f"- Metric: {f.metric.name} = {f.metric.value:.4f}")
        lines.append(f"- 95% CI: [{f.metric.ci_low:.4f}, {f.metric.ci_high:.4f}]")
        lines.append(f"- Hash: {compute_finding_hash(f)}")
        lines.append("")
    return "\n".join(lines)