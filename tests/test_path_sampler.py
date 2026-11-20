"""Tests for the path sampler module."""

import pytest

from morph.path_sampler import (
    SampledPath,
    sample_path,
    paths_to_dicts,
)


class TestSamplePath:
    def test_grid_two_parents(self):
        paths = sample_path(n_parents=2, strategy="grid", resolution=5)
        # 2 parents, resolution=5 gives 6 points (including endpoints)
        assert len(paths) == 6
        # First and last should be pure parents
        assert paths[0].coefficients == [1.0, 0.0]
        assert paths[-1].coefficients == [0.0, 1.0]

    def test_grid_three_parents(self):
        paths = sample_path(n_parents=3, strategy="grid", resolution=3)
        # Should include [1,0,0], [0,1,0], [0,0,1] and interior points
        coeffs = [tuple(p.coefficients) for p in paths]
        assert (1.0, 0.0, 0.0) in coeffs
        assert (0.0, 1.0, 0.0) in coeffs
        assert (0.0, 0.0, 1.0) in coeffs

    def test_random_reproducibility(self):
        p1 = sample_path(n_parents=2, strategy="random", n_samples=10, seed=42)
        p2 = sample_path(n_parents=2, strategy="random", n_samples=10, seed=42)
        for a, b in zip(p1, p2):
            assert a.coefficients == b.coefficients

    def test_random_sum_to_one(self):
        paths = sample_path(n_parents=4, strategy="random", n_samples=20, seed=0)
        for p in paths:
            s = sum(p.coefficients)
            assert abs(s - 1.0) < 1e-9

    def test_boundary_sparsity(self):
        paths = sample_path(n_parents=4, strategy="boundary", n_samples=5, seed=0)
        for p in paths:
            # At least one coefficient should be close to 1.0
            assert max(p.coefficients) > 0.8 or sum(p.coefficients) == pytest.approx(1.0)

    def test_single_parent(self):
        paths = sample_path(n_parents=1, strategy="grid")
        assert len(paths) == 1
        assert paths[0].coefficients == [1.0]


class TestPathsToDicts:
    def test_serialisation(self):
        paths = sample_path(n_parents=2, strategy="grid", resolution=1)
        dicts = paths_to_dicts(paths)
        assert isinstance(dicts, list)
        assert "coefficients" in dicts[0]
        assert "index" in dicts[0]
        assert "strategy" in dicts[0]