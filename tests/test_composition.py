"""Tests for the composition adapter module."""

import copy

import numpy as np
import pytest

from morph.composition import (
    linear_interpolation,
    compose,
    validate_compatibility,
    copy_state_dict,
)
from morph.config import MergeRecipe


def make_test_state_dict(shape: tuple = (10, 20), seed: int = 0):
    rng = np.random.default_rng(seed)
    return {"weight": rng.random(shape).astype(np.float32), "bias": rng.random(20).astype(np.float32)}


class TestValidateCompatibility:
    def test_compatible_dicts_pass(self):
        sd1 = make_test_state_dict()
        sd2 = make_test_state_dict()
        keys = validate_compatibility([sd1, sd2])
        assert "weight" in keys
        assert "bias" in keys

    def test_missing_key_raises(self):
        sd1 = {"a": np.array([1.0])}
        sd2 = {"b": np.array([1.0])}
        with pytest.raises(ValueError):
            validate_compatibility([sd1, sd2])

    def test_shape_mismatch_raises(self):
        sd1 = {"a": np.array([1.0])}
        sd2 = {"a": np.array([1.0, 2.0])}
        with pytest.raises(ValueError):
            validate_compatibility([sd1, sd2])


class TestLinearInterpolation:
    def test_midpoint(self):
        sd1 = {"a": np.array([0.0, 0.0])}
        sd2 = {"a": np.array([1.0, 1.0])}
        result = linear_interpolation([sd1, sd2], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result["a"], np.array([0.5, 0.5]))

    def test_weights(self):
        sd1 = {"a": np.array([10.0])}
        sd2 = {"a": np.array([0.0])}
        result = linear_interpolation([sd1, sd2], [0.3, 0.7])
        np.testing.assert_almost_equal(result["a"], 3.0)

    def test_three_parents(self):
        sd1 = {"a": np.array([1.0])}
        sd2 = {"a": np.array([0.0])}
        sd3 = {"a": np.array([0.0])}
        result = linear_interpolation([sd1, sd2, sd3], [0.5, 0.3, 0.2])
        np.testing.assert_almost_equal(result["a"], 0.5)

    def test_coefficients_invalid_sum(self):
        sd1 = make_test_state_dict()
        sd2 = make_test_state_dict()
        with pytest.raises(ValueError):
            linear_interpolation([sd1, sd2], [0.7, 0.5])

    def test_coefficients_negative(self):
        sd1 = make_test_state_dict()
        sd2 = make_test_state_dict()
        with pytest.raises(ValueError):
            linear_interpolation([sd1, sd2], [1.1, -0.1])


class TestCompose:
    def test_dispatch_linear(self):
        sd1 = make_test_state_dict()
        sd2 = make_test_state_dict()
        recipe = MergeRecipe(
            method="linear_interpolation",
            parents=["a", "b"],
            coefficients=[0.5, 0.5],
        )
        result = compose(recipe, [sd1, sd2])
        assert "weight" in result

    def test_unsupported_method_raises(self):
        recipe = MergeRecipe(method="unsupported", parents=["a"], coefficients=[1.0])
        with pytest.raises(ValueError):
            compose(recipe, [{}])


class TestCopyStateDict:
    def test_deep_copy(self):
        original = {"a": np.array([1.0, 2.0])}
        copy_sd = copy_state_dict(original)
        original["a"][0] = 99.0
        assert copy_sd["a"][0] == 1.0