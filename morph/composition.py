"""Composition adapter.

Applies a supported merge method to parent model weights and validates
compatibility.  The first supported method is *linear interpolation* of
compatible state dictionaries:

    theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B

The merge operates on numpy arrays internally so the core logic is
framework-agnostic; PyTorch state dicts are converted on the way in and
back out.  This keeps the pure-merge logic unit-testable without a GPU or
torch installed.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .config import MergeRecipe


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a torch tensor / numpy array / python scalar to numpy."""
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "detach"):  # torch.Tensor
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):  # e.g. jax
        return np.asarray(t.numpy())
    return np.asarray(t)


def _from_numpy(arr: np.ndarray, template: Any) -> Any:
    """Restore the original framework type of `template` from a numpy array."""
    if hasattr(template, "detach"):  # torch.Tensor
        import torch

        return torch.from_numpy(arr).to(template.dtype).to(template.device)
    if hasattr(template, "numpy"):
        return type(template)(arr)
    return arr


def validate_compatibility(state_dicts: list[dict[str, Any]]) -> list[str]:
    """Return the shared parameter keys and raise if shapes disagree.

    All state dicts must have exactly the same keys with the same shapes
    for a weight-space merge to be meaningful.
    """
    if len(state_dicts) < 2:
        raise ValueError("need at least two parent state dicts to merge")
    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != set(keys):
            missing = set(keys) - set(sd.keys())
            extra = set(sd.keys()) - set(keys)
            raise ValueError(
                "state dict key mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        for k in keys:
            a = _to_numpy(state_dicts[0][k])
            b = _to_numpy(sd[k])
            if a.shape != b.shape:
                raise ValueError(
                    f"shape mismatch for {k!r}: {a.shape} vs {b.shape}"
                )
    return keys


def _check_shape_compatibility(arrays: list[np.ndarray]) -> None:
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f"incompatible shapes for interpolation: {sorted(shapes)}")


def linear_interpolation(
    state_dicts: list[dict[str, Any]],
    coefficients: list[float],
) -> dict[str, Any]:
    """Convex linear interpolation of compatible state dicts.

    theta(alpha) = sum_i alpha_i * theta_i, with sum_i alpha_i = 1.
    """
    if len(state_dicts) != len(coefficients):
        raise ValueError(
            "coefficients must have one entry per parent state dict"
        )
    if len(state_dicts) < 2:
        raise ValueError("linear_interpolation needs at least two parents")
    if abs(sum(coefficients) - 1.0) > 1e-9:
        raise ValueError("coefficients must sum to 1.0 for interpolation")
    if any(c < 0 for c in coefficients):
        raise ValueError("coefficients must be non-negative")

    keys = validate_compatibility(state_dicts)
    merged: dict[str, Any] = {}
    for k in keys:
        arrays = [_to_numpy(sd[k]) for sd in state_dicts]
        _check_shape_compatibility(arrays)
        result = np.zeros_like(arrays[0])
        for arr, c in zip(arrays, coefficients):
            result = result + c * arr
        merged[k] = _from_numpy(result, state_dicts[0][k])
    return merged


def compose(recipe: MergeRecipe, state_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Dispatch to the merge method declared in `recipe`.

    Returns the composed state dict.  Raises on unsupported methods or
    incompatible parents.
    """
    recipe.validate()
    if len(state_dicts) != len(recipe.parents):
        raise ValueError(
            "number of parent state dicts does not match recipe parents"
        )

    method = recipe.method
    if method == "linear_interpolation":
        return linear_interpolation(state_dicts, recipe.coefficients)
    raise ValueError(f"unsupported merge method: {method!r}")


def copy_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a state dict so callers cannot mutate the registry's copy."""
    return copy.deepcopy(state_dict)