"""Tests for the probe engine module."""

import pytest

from morph.probes import (
    ProbeSpec,
    ProbeSuite,
    ProbeResult,
    run_probe_suite,
)


class TestProbeSpec:
    def test_basic_spec(self):
        spec = ProbeSpec(name="sentiment-test", version=1, inputs=["input text"])
        assert spec.name == "sentiment-test"
        assert spec.version == 1

    def test_to_dict(self):
        spec = ProbeSpec(name="test", expected_label="positive")
        d = spec.to_dict()
        assert d["name"] == "test"
        assert d["expected_label"] == "positive"


class TestProbeSuite:
    def test_empty_suite(self):
        suite = ProbeSuite()
        assert len(suite) == 0
        assert suite.version >= 0

    def test_version_consistency(self):
        suite = ProbeSuite()
        suite.add(ProbeSpec(name="probe-a", version=1))
        v1 = suite.version
        suite.add(ProbeSpec(name="probe-a", version=2))
        v2 = suite.version
        assert v1 != v2

    def test_version_changes_with_content(self):
        suite = ProbeSuite()
        suite.add(ProbeSpec(name="probe", version=1))
        v1 = suite.version
        suite2 = ProbeSuite()
        suite2.add(ProbeSpec(name="different", version=1))
        assert suite.version != suite2.version


class TestProbeResult:
    def test_to_dict(self):
        result = ProbeResult(
            probe_name="test",
            probe_version=1,
            model_id="model-a",
            input="hello",
            output="positive",
        )
        d = result.to_dict()
        assert d["probe_name"] == "test"
        assert d["model_id"] == "model-a"
        assert d["output"] == "positive"


class TestRunProbeSuite:
    def test_single_probe(self):
        suite = ProbeSuite([ProbeSpec(name="test", version=1, inputs=["input"])])
        pred = lambda x: "positive"
        results = run_probe_suite(suite, "model-a", pred)
        assert len(results) == 1
        assert results[0].model_id == "model-a"
        assert results[0].output == "positive"

    def test_multiple_probes(self):
        suite = ProbeSuite(
            [
                ProbeSpec(name="test", version=1, inputs=["input1"]),
                ProbeSpec(name="test2", version=1, inputs=["input2"]),
            ]
        )
        pred = lambda x: "label"
        results = run_probe_suite(suite, "model-b", pred)
        assert len(results) == 2

    def test_tuple_return_handling(self):
        suite = ProbeSuite([ProbeSpec(name="test", version=1, inputs=["input"])])
        pred = lambda x: ("label", [0.1, 0.9])
        results = run_probe_suite(suite, "model-a", pred, log_logits=True)
        assert results[0].output == "label"
        assert results[0].logits == [0.1, 0.9]


def test_run_probe_suite_integration():
    """Integration test: run probes against a mock model."""
    suite = ProbeSuite(
        [
            ProbeSpec(name="probe-1", version=1, inputs=["What is 2+2?"]),
            ProbeSpec(name="probe-2", version=1, inputs=["Who is Shakespeare?"]),
        ]
    )
    # Mock model that returns the input prefixed
    def mock_predict(x: str):
        return x.upper()

    results = run_probe_suite(suite, "mock-model", mock_predict)
    assert len(results) == 2
    assert results[0].output == "WHAT IS 2+2?"
    assert results[1].output == "WHO IS SHAKESPEARE?"