"""Tests for the model registry module."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from morph.registry import ModelRegistry, ModelRecord, sha256_file


class TestSHA256:
    def test_sha256_file_returns_hex(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            f.flush()
            result = sha256_file(f.name)
        assert len(result) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_file_consistency(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data for hashing")
            f.flush()
            r1 = sha256_file(f.name)
            r2 = sha256_file(f.name)
        assert r1 == r2
        expected = hashlib.sha256(b"test data for hashing").hexdigest()
        assert r1 == expected


class TestModelRecord:
    def test_to_dict(self):
        r = ModelRecord(
            model_id="test-model-v1",
            source="huggingface",
            revision="abc123",
            architecture="BERT-base",
            tokenizer="wordpiece",
            licence="apache-2.0",
            file_path="/models/test.pt",
            file_hash="deadbeef",
            parameter_count=110_000_000,
        )
        d = r.to_dict()
        assert d["model_id"] == "test-model-v1"
        assert d["parameter_count"] == 110_000_000


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        rec = reg.register(
            "bert-base-uncased",
            source="huggingface",
            revision="main",
            architecture="bert-base-uncased",
            tokenizer="bert-base-uncased",
            licence="apache-2.0",
        )
        assert rec.model_id == "bert-base-uncased"
        assert "bert-base-uncased" in reg
        got = reg.get("bert-base-uncased")
        assert got is rec

    def test_get_missing_raises(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_manifest_structure(self):
        reg = ModelRegistry()
        reg.register("model-a", architecture="bert")
        reg.register("model-b", architecture="roberta")
        manifest = reg.manifest()
        assert manifest["count"] == 2
        assert len(manifest["models"]) == 2

    def test_save_manifest(self):
        reg = ModelRegistry()
        reg.register("test-model", architecture="test")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        saved = reg.save_manifest(path)
        assert saved.exists()
        content = json.loads(Path(path).read_text())
        assert content["count"] == 1
        Path(path).unlink()