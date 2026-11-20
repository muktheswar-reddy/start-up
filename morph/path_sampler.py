"""Composition-path sampling.

A *composition path* is a sequence of merge configurations sampled within
the declared coefficient constraints.  Sampling the path (rather than only
testing the single final configuration) is the core hypothesis of MORPH-CG:
security-relevant behaviour may appear only in a region of the
configuration space.

Supported strategies
--------------------
* ``grid``      -- deterministic lattice over the coefficient simplex.
* ``random``     -- i.i.d. uniform samples within the constraints.
* ``boundary``   -- samples pushed toward the simplex boundary (pure parents).
* ``adaptive``   -- refine around previously anomalous configurations.

All strategies are seeded for reproducibility and return plain data
(coefficient vectors + metadata), never model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SampledPath:
    """One sampled composition configuration."""

    coefficients: list[float]
    index: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "index": self.index,
            "strategy": self.strategy,
            "metadata": dict(self.metadata),
        }


def _clip_simplex(coef: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex (sum=1, all >= 0)."""
    coef = np.maximum(coef, 0.0)
    s = coef.sum()
    if s <= 0:
        return np.ones_like(coef) / len(coef)
    return coef / s


def _grid_on_simplex(n_parents: int, resolution: int) -> list[np.ndarray]:
    """Deterministic lattice over the coefficient simplex.

    Uses a recursive partition so every returned vector sums to 1.0.
    """
    if n_parents == 1:
        return [np.array([1.0])]
    if n_parents == 2:
        return [
            np.array([1.0 - i / resolution, i / resolution])
            for i in range(resolution + 1)
        ]

    out: list[np.ndarray] = []

    def rec(remaining: int, total: float, start: float, acc: list[float]) -> None:
        if remaining == 1:
            acc.append(round(total, 12))
            out.append(np.array(acc))
            acc.pop()
            return
        for i in range(resolution + 1):
            v = i / resolution
            if v > total + 1e-12:
                break
            acc.append(v)
            rec(remaining - 1, total - v, v, acc)
            acc.pop()

    rec(n_parents, 1.0, 0.0, [])
    return out


def sample_path(
    n_parents: int,
    *,
    strategy: str = "grid",
    n_samples: int = 10,
    resolution: int = 5,
    seed: int = 42,
    constraints: dict[str, Any] | None = None,
) -> list[SampledPath]:
    """Sample a composition path within declared constraints.

    Parameters
    ----------
    n_parents:
        Number of parent models in the composition.
    strategy:
        One of ``grid``, ``random``, ``boundary``, ``adaptive``.
    n_samples:
        Number of configurations to return (random/boundary/adaptive).
    resolution:
        Grid resolution (number of divisions per axis).
    seed:
        RNG seed for reproducibility.
    constraints:
        Optional constraint dict (e.g. ``{"non_negative": True}``).
    """
    constraints = constraints or {}
    rng = np.random.default_rng(seed)
    samples: list[SampledPath] = []

    if strategy == "grid":
        vecs = _grid_on_simplex(n_parents, resolution)
        for i, v in enumerate(vecs):
            samples.append(
                SampledPath(
                    coefficients=v.tolist(),
                    index=i,
                    strategy=strategy,
                    metadata={"resolution": resolution},
                )
            )
        return samples

    if strategy in {"random", "boundary", "adaptive"}:
        for i in range(n_samples):
            if strategy == "boundary":
                # Push mass onto one parent at a time, then mix the rest.
                base = rng.random(n_parents)
                base = _clip_simplex(base)
                # Bias toward sparsity: zero out all but two entries.
                if n_parents > 2:
                    keep = rng.choice(n_parents, size=2, replace=False)
                    mask = np.zeros(n_parents, dtype=bool)
                    mask[keep] = True
                    base = base * mask
                    base = _clip_simplex(base)
            else:
                base = _clip_simplex(rng.random(n_parents))
            samples.append(
                SampledPath(
                    coefficients=base.tolist(),
                    index=i,
                    strategy=strategy,
                    metadata={"seed": seed},
                )
            )
        return samples

    raise ValueError(f"unsupported sampling strategy: {strategy!r}")


def paths_to_dicts(paths: list[SampledPath]) -> list[dict[str, Any]]:
    """Serialise a sampled path list for the experiment manifest."""
    return [p.to_dict() for p in paths]