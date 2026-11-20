"""Tests for the reporter module."""

import json
import tempfile
from pathlib import Path

import pytest

from morph.reporting import EvidenceBundle, generate_research_memo
from morph.config import MergeRecipe
from morph.registry import ModelRegistry
from morph.path_sampler import SampledPath
from morph.metrics import MetricResult


class TestEvidenceBundle:
    def test_empty_bundle(self):
        registry = ModelRegistry()
        recipe = MergeRecipe()
        bundle = EvidenceBundle(
            registry=registry,
            recipe=recipe,
            paths=[],
            base_metrics=[],
            findings=[],
        )
        d = bundle.to_dict()
        assert "manifest_version" in d
        assert d["summary"]["total_paths"] == 0

    def test_to_json_roundtrip(self):
        registry = ModelRegistry()
        registry.register("test-model", architecture="test")
        recipe = MergeRecipe(
            method="linear_interpolation",
            parents=["a", "b"],
            coefficients=[0.5, 0.5],
        )
        bundle = EvidenceBundle(
            registry=registry,
            recipe=recipe,
            paths=[SampledPath(coefficients=[0.5, 0.5], index=0, strategy="grid")],
            base_metrics=[],
            findings=[],
        )
        j = bundle.to_json()
        parsed = json.loads(j)
        assert parsed["registry_manifest"]["count"] == 1

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.json"
            registry = ModelRegistry()
            registry.register("test", architecture="test")
            bundle = EvidenceBundle(
                registry=registry,
                recipe=MergeRecipe(),
                paths=[],
                base_metrics=[],
                findings=[],
            )
            saved = bundle.save(path)
            assert saved.exists()
            content = json.loads(path.read_text())
            assert "generated_at" in content


class TestGenerateResearchMemo:
    def test_memo_content(self):
        registry = ModelRegistry()
        registry.register("model-a", architecture="bert")
        recipe = MergeRecipe(
            method="linear_interpolation",
            parents=["model-a"],
            coefficients=[1.0],
        )
        bundle = EvidenceBundle(
            registry=registry,
            recipe=recipe,
            paths=[SampledPath(coefficients=[0.5, 0.5], index=0, strategy="grid")],
            base_metrics=[],
            findings=[],
        )
        memo = generate_research_memo(bundle)
        assert "MORPH-CG Research Memo" in memo
        assert "bert" in memo
        assert "linear_interpolation" in memo