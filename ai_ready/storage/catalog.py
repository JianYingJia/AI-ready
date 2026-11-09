"""Parquet/DuckDB catalog plus content-addressed snapshots."""

import hashlib
import json
from pathlib import Path

from ai_ready.utils.io import write_json


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dataset_version(source_hashes: list[str], config: dict) -> str:
    payload = json.dumps(
        {"sources": sorted(source_hashes), "config": config}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_duckdb(output_dir: Path) -> Path:
    import duckdb

    database = output_dir / "catalog.duckdb"
    tables = {path.stem: path for path in output_dir.glob("*.parquet") if path.name not in {"embeddings.parquet"}}
    connection = duckdb.connect(str(database))
    try:
        for name, path in sorted(tables.items()):
            escaped = path.resolve().as_posix().replace("'", "''")
            connection.execute(f"CREATE OR REPLACE VIEW \"{name}\" AS SELECT * FROM read_parquet('{escaped}')")
        connection.execute("CREATE OR REPLACE TABLE _catalog_info(name VARCHAR, path VARCHAR)")
        connection.executemany(
            "INSERT INTO _catalog_info VALUES (?, ?)",
            [(name, str(path.resolve())) for name, path in sorted(tables.items())],
        )
    finally:
        connection.close()
    return database


def write_snapshot(output_dir: Path, version: str, config_sha256: str) -> dict:
    # manifest.json is finalized after publication; its processing config is
    # represented separately by config_sha256.
    ignored = {"snapshot.json", "catalog.duckdb", "manifest.json"}
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in ignored or "mlruns" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    snapshot = {
        "schema_version": 1,
        "dataset_version": version,
        "config_sha256": config_sha256,
        "files": files,
    }
    write_json(output_dir / "snapshot.json", snapshot)
    return snapshot


class RunTracker:
    def __init__(self, enabled: bool, tracking_dir: Path, experiment: str):
        self.enabled = enabled
        self.tracking_dir = tracking_dir
        self.experiment = experiment
        self.run = None

    def start(self, run_name: str) -> str | None:
        if not self.enabled:
            return None
        import mlflow

        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        database_uri = "sqlite:///" + (self.tracking_dir / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(database_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment, artifact_location=(self.tracking_dir / "artifacts").resolve().as_uri()
            )
        mlflow.set_experiment(self.experiment)
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"layer": "L1-L2", "dataset_version": run_name})
        return self.run.info.run_id

    def log_stage(self, stage: str, summary: dict) -> None:
        if not self.run:
            return
        import mlflow

        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(f"{stage}.{key}", float(value))

    def finish(self, status: str, manifest: Path) -> None:
        if not self.run:
            return
        import mlflow

        mlflow.set_tag("pipeline_status", status)
        if manifest.is_file():
            mlflow.log_artifact(str(manifest), artifact_path="manifests")
        mlflow.end_run(status="FINISHED" if status in {"completed", "partial"} else "FAILED")
        self.run = None
