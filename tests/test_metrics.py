"""Tests for the behaviour comparator / metrics module."""

import numpy as np
import pytest

from morph.metrics import (
    MetricResult,
    label_agreement_rate,
    label_flip_rate,
    kl_divergence,
    js_divergence,
    distribution_distance,
    confidence_interval,
)


class TestMetricResult:
    def test_to_dict(self):
        m = MetricResult(name="test", value=0.5, ci_low=0.3, ci_high=0.7)
        d = m.to_dict()
        assert d["value"] == 0.5
        assert d["ci_low"] == 0.3


class TestLabelAgreement:
    def test_perfect_agreement(self):
        r = label_agreement_rate(["a", "b", "a"], ["a", "b", "a"])
        assert r.value == 1.0

    def test_zero_agreement(self):
        r = label_agreement_rate(["a", "b"], ["c", "d"])
        assert r.value == 0.0

    def test_partial_agreement(self):
        r = label_agreement_rate(["a", "b", "c"], ["a", "x", "c"])
        assert abs(r.value - 2 / 3) < 1e-9


class TestLabelFlipRate:
    def test_no_flips(self):
        r = label_flip_rate(["a", "b", "c"], ["a", "b", "c"])
        assert r.value == 0.0

    def test_all_flips(self):
        r = label_flip_rate(["a", "b"], ["c", "d"])
        assert r.value == 1.0

    def test_half_flips(self):
        r = label_flip_rate(["a", "b", "c", "d"], ["x", "b", "y", "d"])
        assert r.value == 0.5


class TestDistributionDistance:
    def test_js_identical(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        d = js_divergence(p, q)
        assert d == 0.0

    def test_js_different(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        d = js_divergence(p, q)
        assert d > 0

    def test_distribution_distance_wrapper(self):
        obs = [np.array([1, 0]), np.array([0.5, 0.5])]
        ref = [np.array([0.5, 0.5]), np.array([1, 0])]
        r = distribution_distance(obs, ref)
        assert r.value > 0


class TestConfidenceInterval:
    def test_single_value(self):
        lo, hi = confidence_interval([5.0])
        assert lo == hi == 5.0

    def test_multiple_values(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = confidence_interval(vals, confidence=0.95)
        mean = np.mean(vals)
        assert lo <= mean <= hi

    def test_uniform_values(self):
        lo, hi = confidence_interval([3.14, 3.14, 3.14])
        assert lo == hi == 3.14