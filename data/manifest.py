"""
Content-addressed data manifest.

A manifest is a JSON file listing the training-data shards by content hash.
The canonical training loop (recipe/train.py) verifies the hash of each shard
before consuming it. In the Phase 0.5+ proof-test Docker, the manifest hash
is extended into a TDX RTMR so the attestation chain proves which data the
training actually saw — closing the audit-reproducibility gap.

For Phase 0 the manifest is just a JSON file colocated with the canonical
recipe. Replace with on-chain commitment in Phase 0.5+.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


CHUNK_BYTES = 1 << 20  # 1 MiB streaming chunks for hashing large shards


def shard_hash(path: Path | str) -> str:
    """SHA-256 of a shard file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


@dataclass
class ShardEntry:
    relpath: str
    sha256: str
    n_tokens: int
    bytes: int


@dataclass
class DataManifest:
    track: str
    tokenizer: str
    vocab_size: int
    dtype: str
    shards: list[ShardEntry] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(s.n_tokens for s in self.shards)

    def manifest_hash(self) -> str:
        """Deterministic hash of the manifest itself — the value extended into
        the proof-test attestation user_data so validators can verify which
        manifest the miner trained against."""
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DataManifest":
        d = json.loads(text)
        shards = [ShardEntry(**s) for s in d.pop("shards", [])]
        return cls(shards=shards, **d)

    @classmethod
    def from_path(cls, path: Path | str) -> "DataManifest":
        return cls.from_json(Path(path).read_text())

    def write(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json())


def build_manifest(
    track: str,
    tokenizer: str,
    vocab_size: int,
    dtype: str,
    shards: Iterable[Path],
    base_dir: Path,
) -> DataManifest:
    import numpy as np

    entries: list[ShardEntry] = []
    np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[dtype]
    bytes_per_token = np_dtype().itemsize
    for shard_path in shards:
        shard_path = Path(shard_path)
        size = shard_path.stat().st_size
        n_tokens = size // bytes_per_token
        entries.append(
            ShardEntry(
                relpath=str(shard_path.relative_to(base_dir)),
                sha256=shard_hash(shard_path),
                n_tokens=n_tokens,
                bytes=size,
            )
        )
    return DataManifest(
        track=track,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        dtype=dtype,
        shards=entries,
    )


def verify_manifest(manifest: DataManifest, base_dir: Path | str) -> list[str]:
    """Return a list of mismatched shard paths. Empty list = all good."""
    base = Path(base_dir)
    bad: list[str] = []
    for entry in manifest.shards:
        path = base / entry.relpath
        if not path.exists():
            bad.append(f"missing: {entry.relpath}")
            continue
        h = shard_hash(path)
        if h != entry.sha256:
            bad.append(f"hash mismatch: {entry.relpath} (expected {entry.sha256[:8]}, got {h[:8]})")
    return bad
