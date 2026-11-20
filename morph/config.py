"""Configuration dataclasses for MORPH-CG experiments.

These types describe *what* an experiment does.  They are deliberately plain
data so they can be serialised to/from YAML and embedded in reproducible
experiment manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MergeRecipe:
    """A declared composition recipe.

    The recipe is the contract: it says which merge method is used, which
    parents participate, and what coefficient constraints apply.  Sampling
    happens *within* these declared constraints, never outside them.
    """

    method: str = "linear_interpolation"
    parents: list[str] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Check that the recipe is internally consistent."""
        if not self.parents:
            raise ValueError("MergeRecipe requires at least one parent model")
        if len(self.parents) != len(self.coefficients):
            raise ValueError(
                "coefficients must have one entry per parent "
                f"({len(self.parents)} parents, {len(self.coefficients)} coefficients)"
            )
        constraint = self.constraints.get("sum_to_one", False)
        if constraint and abs(sum(self.coefficients) - 1.0) > 1e-9:
            raise ValueError("coefficients must sum to 1.0 when sum_to_one is set")
        if any(c < 0 for c in self.coefficients) and self.constraints.get("non_negative", False):
            raise ValueError("coefficients must be non-negative when non_negative is set")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeSpec:
    """A single probe definition: a named, versioned test input."""

    name: str
    version: int = 1
    description: str = ""
    inputs: list[Any] = field(default_factory=list)
    expected_label: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_id: str = "morph-cg-v0.1"
    seed: int = 42
    device: str = "cpu"
    parents: list[str] = field(default_factory=list)
    recipe: MergeRecipe = field(default_factory=MergeRecipe)
    probes: list[ProbeSpec] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)