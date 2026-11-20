"""Tests for the attribution engine module."""

import pytest

from morph.attribution import (
    AttributionFinding,
    estimate_nonadditive_effect,
    shapley_decomposition,
    coefficient_perturbation,
)
from morph.metrics import MetricResult


class TestAttributionFinding:
    def test_to_dict(self):
        f = AttributionFinding(
            composition_index=0,
            component_index=1,
            metric="flip_rate",
            original_value=0.1,
            modified_value=0.05,
            attribution_score=-0.5,
            uncertainty=(0.08, 0.12),
        )
        d = f.to_dict()
        assert d["composition_index"] == 0
        assert d["component_index"] == 1
        assert d["attribution_score"] == -0.5


class TestEstimateNonadditive:
    def test_zero_effect(self):
        child = MetricResult("metric", 0.5, 0.4, 0.6)
        parents = [
            MetricResult("metric", 0.25, 0.2, 0.3),
            MetricResult("metric", 0.25, 0.2, 0.3),
        ]
        effect = estimate_nonadditive_effect(child, parents)
        assert abs(effect) < 1e-9  # 0.5 - 0.5 = 0

    def test_positive_emergence(self):
        child = MetricResult("metric", 0.9, 0.8, 1.0)
        parents = [
            MetricResult("metric", 0.3, 0.2, 0.4),
            MetricResult("metric", 0.2, 0.1, 0.3),
        ]
        effect = estimate_nonadditive_effect(child, parents)
        assert effect > 0  # 0.9 > 0.5


class TestShapleyDecomposition:
    def test_three_components(self):
        parent_values = {"model-a": 0.2, "model-b": 0.3, "model-c": 0.1}
        composed = 0.6
        scores = shapley_decomposition(parent_values, composed, "metric")
        assert scores["model-a"] > 0
        assert scores["model-b"] > 0
        assert scores["model-c"] > 0


class TestCoefficientPerturbation:
    def test_positive_perturbation(self):
        original = MetricResult("metric", 0.1, 0.05, 0.15)
        perturbed = MetricResult("metric", 0.15, 0.1, 0.2)
        finding = coefficient_perturbation(original, perturbed, coefficient_delta=0.05)
        assert finding.attribution_score > 0  # Increased


def test_estimate_nonadditive_edge_case():
    """Test with zero parents returns negative delta."""
    child = MetricResult("metric", 0.1, 0.05, 0.15)
    effect = estimate_nonadditive_effect(child, [])
    assert effect == 0.1  # 0.1 - 0 = 0.1