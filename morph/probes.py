"""Probe engine.

A *probe* is a fixed, versioned test input run against parent and composed
models under identical conditions.  Probes are versioned so that changing
the probe suite is a deliberate, auditable event in the experiment manifest.

The probe engine is model-agnostic: it accepts a callable ``predict`` that
maps a text input to a label (and optionally logits), which lets the same
probe suite run against any model family.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from .config import ProbeSpec


@dataclass
class ProbeResult:
    """Output of running one probe against one model."""

    probe_name: str
    probe_version: int
    model_id: str
    input: Any
    output: Any
    logits: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "probe_version": self.probe_version,
            "model_id": self.model_id,
            "input": _serialisable(self.input),
            "output": _serialisable(self.output),
            "logits": _serialisable(self.logits),
            "metadata": dict(self.metadata),
        }


def _serialisable(obj: Any) -> Any:
    """Make an object JSON-friendly; numpy/torch scalars become floats."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)


class ProbeSuite:
    """A fixed, versioned collection of probes."""

    def __init__(self, probes: list[ProbeSpec] | None = None) -> None:
        self._probes: list[ProbeSpec] = list(probes or [])

    @property
    def version(self) -> int:
        """Suite version = hash of the sorted probe names+versions."""
        h = hashlib.sha256()
        for p in sorted(self._probes, key=lambda x: x.name):
            h.update(f"{p.name}@{p.version}".encode())
        return int(h.hexdigest()[:8], 16)

    def add(self, probe: ProbeSpec) -> None:
        self._probes.append(probe)

    def __len__(self) -> int:
        return len(self._probes)

    def __iter__(self):
        return iter(self._probes)

    def all(self) -> list[ProbeSpec]:
        return list(self._probes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "probes": [p.to_dict() for p in self._probes],
        }


def run_probe_suite(
    suite: ProbeSuite,
    model_id: str,
    predict: Callable[[Any], Any],
    *,
    log_logits: bool = False,
) -> list[ProbeResult]:
    """Run every probe in `suite` against one model.

    `predict` is a callable that takes a probe input and returns either a
    label or a tuple ``(label, logits)``.  Keeping the model-loading logic
    outside the probe engine lets the suite run against any backend.
    """
    results: list[ProbeResult] = []
    for probe in suite:
        raw = predict(probe.inputs[0] if probe.inputs else None)
        logits = None
        output = raw
        if isinstance(raw, tuple) and len(raw) == 2:
            output, logits = raw
        results.append(
            ProbeResult(
                probe_name=probe.name,
                probe_version=probe.version,
                model_id=model_id,
                input=probe.inputs[0] if probe.inputs else None,
                output=output,
                logits=logits if log_logits else None,
                metadata={"expected_label": probe.expected_label},
            )
        )
    return results