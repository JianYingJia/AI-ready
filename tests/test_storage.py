import duckdb
import pyarrow.parquet as pq

from ai_ready.storage.catalog import build_duckdb, write_snapshot
from ai_ready.utils.io import write_parquet


def test_versioned_parquet_duckdb_and_snapshot(tmp_path):
    write_parquet(
        tmp_path / "documents.parquet",
        [{"dataset_version": "v1", "document_id": "doc", "permissions": ["legal"]}],
        metadata={"dataset_version": "v1"},
    )
    metadata = pq.read_metadata(tmp_path / "documents.parquet").metadata
    assert metadata[b"dataset_version"] == b"v1"
    database = build_duckdb(tmp_path)
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("select document_id from documents").fetchone()[0] == "doc"
    snapshot = write_snapshot(tmp_path, "v1", "config-hash")
    assert any(row["path"] == "documents.parquet" for row in snapshot["files"])
    assert all(row["path"] != "catalog.duckdb" for row in snapshot["files"])
