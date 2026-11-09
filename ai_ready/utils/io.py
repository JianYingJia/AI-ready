"""UTF-8, atomic output helpers."""

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_parquet(
    path: Path,
    rows: Iterable[dict],
    *,
    metadata: dict[str, str] | None = None,
    compression: str = "zstd",
) -> None:
    """Write nested records atomically and attach dataset-level metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    values = list(rows)
    if not values:
        raise ValueError(f"Cannot infer a Parquet schema for an empty table: {path.name}")
    table = pa.Table.from_pylist(values)
    if metadata:
        existing = dict(table.schema.metadata or {})
        existing.update({str(key).encode(): str(value).encode() for key, value in metadata.items()})
        table = table.replace_schema_metadata(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression=None if compression == "none" else compression)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()
