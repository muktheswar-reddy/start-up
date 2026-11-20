"""Model registry.

Records model identifiers, source, licence, revision, file hashes,
architecture and tokenizer, and emits an immutable experiment manifest.

The registry never loads model weights.  It only inspects metadata and
hashes, which keeps it cheap and safe (no untrusted pickle loading).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, streaming to bound memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class ModelRecord:
    """A registered model's provenance metadata."""

    model_id: str
    source: str = ""
    revision: str = ""
    architecture: str = ""
    tokenizer: str = ""
    licence: str = ""
    file_path: str = ""
    file_hash: str = ""
    parameter_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """In-memory registry of models with provenance metadata."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}

    def register(
        self,
        model_id: str,
        *,
        source: str = "",
        revision: str = "",
        architecture: str = "",
        tokenizer: str = "",
        licence: str = "",
        file_path: str = "",
        file_hash: str | None = None,
        parameter_count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ModelRecord:
        """Register a model.  If file_path is given, hash the file."""
        if file_path and not file_hash:
            file_hash = sha256_file(file_path)
        record = ModelRecord(
            model_id=model_id,
            source=source,
            revision=revision,
            architecture=architecture,
            tokenizer=tokenizer,
            licence=licence,
            file_path=file_path,
            file_hash=file_hash or "",
            parameter_count=parameter_count,
            extra=extra or {},
        )
        self._records[model_id] = record
        return record

    def get(self, model_id: str) -> ModelRecord:
        if model_id not in self._records:
            raise KeyError(f"model not registered: {model_id!r}")
        return self._records[model_id]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[ModelRecord]:
        return list(self._records.values())

    def manifest(self) -> dict[str, Any]:
        """Produce an immutable experiment manifest snapshot."""
        return {
            "models": [r.to_dict() for r in self._records.values()],
            "count": len(self._records),
        }

    def save_manifest(self, path: str | Path) -> Path:
        """Write the manifest as JSON to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True))
        return p