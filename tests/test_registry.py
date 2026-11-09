from pathlib import Path

import pytest

from ai_ready.config import RegistryConfig
from ai_ready.core.registry import DocumentRegistry, relate_versions


def test_registration_preserves_versions_and_duplicate_relations(tmp_path):
    (tmp_path / "v1.txt").write_text("同一内容", encoding="utf-8")
    (tmp_path / "v2.txt").write_text("同一内容", encoding="utf-8")
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        """documents:
  - path: v1.txt
    logical_key: policy/a
    version: '1'
    effective_from: '2025-01-01'
    permissions: [legal]
  - path: v2.txt
    logical_key: policy/a
    version: '2'
    effective_from: '2026-01-01'
    permissions: [legal]
""",
        encoding="utf-8",
    )
    registry = DocumentRegistry(tmp_path, RegistryConfig(metadata_file=metadata))
    first = registry.register(tmp_path / "v1.txt")
    second = registry.register(tmp_path / "v2.txt")
    relate_versions([first, second], {first.document_version_id: "同一内容", second.document_version_id: "同一内容"}, 3)
    assert first.document_id == second.document_id
    assert first.document_version_id != second.document_version_id
    assert second.previous_version_id == first.document_version_id
    assert second.exact_duplicate_of == first.document_version_id
    assert first.permissions == ["legal"]


def test_default_registration_uses_relative_path_as_logical_key(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("内容", encoding="utf-8")
    record = DocumentRegistry(tmp_path, RegistryConfig()).register(path)
    assert record.logical_key == Path("policy.txt").as_posix()
    assert record.document_id
    assert record.document_version_id


def test_registration_rejects_empty_permissions(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("内容", encoding="utf-8")
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("documents:\n  - path: policy.txt\n    permissions: []\n", encoding="utf-8")
    registry = DocumentRegistry(tmp_path, RegistryConfig(metadata_file=metadata))
    with pytest.raises(ValueError, match="permissions"):
        registry.register(path)
