"""Document registration and version/duplicate relations."""

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ai_ready.config import RegistryConfig
from ai_ready.core.schema import DocumentRecord, stable_id


class DocumentRegistry:
    def __init__(self, input_dir: Path, config: RegistryConfig):
        self.input_dir = input_dir.resolve()
        self.config = config
        self.metadata = self._load_metadata(config.metadata_file)

    @staticmethod
    def _load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None:
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows = data.get("documents", []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("Document metadata must be a list or contain a documents list")
        result = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Every document metadata row requires a string path")
            key = Path(row["path"]).as_posix()
            if key in result:
                raise ValueError(f"Duplicate document metadata path: {key}")
            result[key] = {key: value for key, value in row.items() if key != "path"}
        return result

    def register(self, path: Path) -> DocumentRecord:
        path = path.resolve()
        relative = path.relative_to(self.input_dir).as_posix()
        values = dict(self.metadata.get(relative, {}))
        with path.open("rb") as handle:
            source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        logical_key = str(values.pop("logical_key", relative))
        if not logical_key.strip():
            raise ValueError(f"logical_key cannot be empty: {relative}")
        document_id = stable_id("document", logical_key)
        version = str(values.pop("version", self.config.default_version))
        if not version.strip():
            raise ValueError(f"version cannot be empty: {relative}")
        document_version_id = stable_id("document-version", document_id, version, source_hash)
        known = {
            "title",
            "effective_from",
            "effective_to",
            "status",
            "permissions",
        }
        metadata = {key: values.pop(key) for key in list(values) if key not in known}
        permissions = values.pop("permissions", self.config.default_permissions)
        if (
            not isinstance(permissions, list)
            or not permissions
            or any(not isinstance(item, str) or not item.strip() for item in permissions)
        ):
            raise ValueError(f"permissions must be a nonempty string list: {relative}")
        return DocumentRecord(
            document_id=document_id,
            document_version_id=document_version_id,
            logical_key=logical_key,
            title=str(values.pop("title", path.stem)),
            source_uri=path.as_uri(),
            source_sha256=source_hash,
            file_size=path.stat().st_size,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            version=version,
            effective_from=values.pop("effective_from", None),
            effective_to=values.pop("effective_to", None),
            status=values.pop("status", "unknown"),
            permissions=permissions,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )


def relate_versions(records: list[DocumentRecord], document_text: dict[str, str], max_distance: int) -> None:
    """Annotate relations but never delete a document version."""
    from simhash import Simhash

    by_hash: dict[str, str] = {}
    fingerprints = {}
    for record in records:
        previous = by_hash.setdefault(record.source_sha256, record.document_version_id)
        if previous != record.document_version_id:
            record.exact_duplicate_of = previous
        fingerprint = Simhash(document_text.get(record.document_version_id, ""), f=64)
        record.simhash64 = f"{fingerprint.value:016x}"
        fingerprints[record.document_version_id] = fingerprint
    for index, record in enumerate(records):
        for other in records[index + 1 :]:
            distance = fingerprints[record.document_version_id].distance(fingerprints[other.document_version_id])
            if distance <= max_distance:
                record.near_duplicate_version_ids.append(other.document_version_id)
                other.near_duplicate_version_ids.append(record.document_version_id)
    by_document: dict[str, list[DocumentRecord]] = {}
    for record in records:
        by_document.setdefault(record.document_id, []).append(record)
    for versions in by_document.values():
        versions.sort(key=lambda item: (item.effective_from or "", item.version, item.source_uri))
        for previous, current in zip(versions, versions[1:]):
            current.previous_version_id = previous.document_version_id
