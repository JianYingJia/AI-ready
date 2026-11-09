"""File-based L1-to-L2 stages with explicit, inspectable artifacts."""

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ai_ready.config import EmbeddingConfig, PipelineConfig
from ai_ready.core.schema import DocumentRecord, EvidenceUnit
from ai_ready.utils.io import read_jsonl, write_json, write_jsonl, write_parquet
from ai_ready.utils.runtime import configure_runtime


def package_versions() -> dict:
    versions = {}
    for name in (
        "py-data-juicer",
        "docling",
        "docling-core",
        "marker-pdf",
        "pymupdf",
        "python-docx",
        "sentence-transformers",
        "transformers",
        "torch",
        "faiss-cpu",
        "hdbscan",
        "bertopic",
        "pyarrow",
        "duckdb",
        "mlflow-skinny",
        "simhash",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class L1Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        configure_runtime(config.cache_dir)
        self._tracker = None

    def _new_stage(self, name: str) -> Path:
        directory = self.config.output_dir / name
        if directory.exists():
            raise FileExistsError(f"Stage already exists: {directory}. Use --output-dir for a new run.")
        directory.mkdir(parents=True)
        return directory

    def parse(self) -> dict:
        from ai_ready.core.registry import DocumentRegistry, relate_versions
        from ai_ready.format.documents import (
            SUPPORTED_SUFFIXES,
            DocumentChunker,
            DocumentParser,
        )

        if not self.config.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")
        files = sorted(
            path
            for path in self.config.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith(("~$", "."))
        )
        if not files:
            raise ValueError(f"No supported documents in {self.config.input_dir}")
        parser = DocumentParser(self.config.parsing)
        chunker = DocumentChunker(self.config.parsing, self.config.embedding)
        registry = DocumentRegistry(self.config.input_dir, self.config.registry)
        directory = self._new_stage("parsed")
        records, documents, errors = [], [], []
        document_text = {}
        for path in files:
            try:
                registration = registry.register(path)
                document, backend = parser.convert(path)
                _, units = chunker.chunk(document, path, backend, registration)
                registration.parser = backend
                registration.unit_count = sum(not unit.is_window_copy for unit in units)
                write_json(
                    directory / "documents" / f"{registration.document_version_id}.json", document.export_to_dict()
                )
                if parser.last_attachment:
                    write_json(
                        directory / "attachments" / f"{registration.document_version_id}.marker.json",
                        parser.last_attachment,
                    )
                records.extend(units)
                documents.append(registration)
                document_text[registration.document_version_id] = "\n".join(
                    unit.raw_text for unit in units if not unit.is_window_copy
                )
            except Exception as error:
                errors.append(
                    {"source_uri": path.resolve().as_uri(), "error": str(error), "type": type(error).__name__}
                )
                write_json(directory / "errors.json", errors)
                if not self.config.parsing.continue_on_error:
                    raise
        if not records:
            raise ValueError("All documents failed parsing; see parsed/errors.json")
        relate_versions(documents, document_text, self.config.registry.simhash_max_distance)
        version = self._compute_dataset_version(documents)
        write_jsonl(directory / "records.jsonl", (unit.model_dump(mode="json") for unit in records))
        write_jsonl(directory / "documents.jsonl", (document.model_dump(mode="json") for document in documents))
        write_parquet(
            self.config.output_dir / "documents.parquet",
            ({"dataset_version": version, **document.model_dump(mode="json")} for document in documents),
            metadata={"dataset_version": version, "schema": "documents-v1"},
            compression=self.config.storage.parquet_compression,
        )
        summary = {
            "files_seen": len(files),
            "files_parsed": len(documents),
            "units": len(records),
            "original_units": sum(not unit.is_window_copy for unit in records),
            "window_units": sum(unit.is_window_copy for unit in records),
            "exact_duplicate_files": sum(bool(document.exact_duplicate_of) for document in documents),
            "dataset_version": version,
            "errors": errors,
        }
        write_json(
            directory / "manifest.json",
            {**summary, "documents": [document.model_dump(mode="json") for document in documents]},
        )
        return summary

    def clean(self) -> dict:
        from ai_ready.ops.cleaning import clean_units

        units = self._read_units("parsed")
        cleaned, audit = clean_units(units, self.config.cleaning)
        directory = self._new_stage("cleaned")
        write_jsonl(directory / "records.jsonl", (unit.model_dump(mode="json") for unit in cleaned))
        write_jsonl(directory / "audit.jsonl", audit)
        summary = {
            "input_units": len(units),
            "output_units": len(cleaned),
            "exact_duplicates": sum(row["action"] == "exact_duplicate" for row in audit),
            "filtered": sum(row["action"] == "filtered" for row in audit),
            "near_duplicate_candidates": sum(row["action"] == "near_duplicate_flagged" for row in audit),
        }
        write_json(directory / "manifest.json", summary)
        if not cleaned:
            raise ValueError("Cleaning retained no evidence; inspect cleaned/audit.jsonl")
        return summary

    def analyze(self) -> dict:
        from ai_ready.difficulty.model import document_difficulty

        units = self._read_units("cleaned")
        documents = self._read_documents()
        record_by_version = {record.document_version_id: record for record in documents}
        for unit in units:
            profile = document_difficulty(
                unit, record_by_version.get(unit.document_version_id), self.config.l2.difficulty_weights
            )
            unit.document_difficulty = profile.model_dump(mode="json")
        directory = self._new_stage("analyzed")
        write_jsonl(directory / "records.jsonl", (unit.model_dump(mode="json") for unit in units))
        version = self._dataset_version()
        write_parquet(
            self.config.output_dir / "document_units.parquet",
            ({"dataset_version": version, **unit.model_dump(mode="json")} for unit in units),
            metadata={"dataset_version": version, "schema": "document-units-v1"},
            compression=self.config.storage.parquet_compression,
        )
        level_counts = {
            level: sum(unit.document_difficulty["level"] == level for unit in units if not unit.is_window_copy)
            for level in ("low", "medium", "high")
        }
        summary = {
            "units": len(units),
            "original_units": sum(not unit.is_window_copy for unit in units),
            **{f"difficulty_{key}": value for key, value in level_counts.items()},
        }
        write_json(directory / "manifest.json", {**summary, "weights": self.config.l2.difficulty_weights.model_dump()})
        return summary

    def index(self) -> dict:
        from ai_ready.embedding.sentence_transformer import TextEmbedder
        from ai_ready.retrieval.faiss_index import FaissStore

        directory = self.config.output_dir / "index"
        if directory.exists():
            raise FileExistsError(f"Index already exists: {directory}")
        units = self._read_units("analyzed")
        embedder = TextEmbedder(self.config.embedding)
        vectors = embedder.encode([unit.embedding_text for unit in units])
        store = FaissStore.build(units, vectors, embedder.resolved_config())
        store.save(directory)
        np.save(directory / "embeddings.npy", vectors, allow_pickle=False)
        return store.metadata

    def cluster(self) -> dict:
        from ai_ready.clustering.topics import cluster_texts
        from ai_ready.retrieval.faiss_index import FaissStore, corpus_digest

        index_dir = self.config.output_dir / "index"
        store = FaissStore.load(index_dir)
        if corpus_digest(self._read_units("analyzed")) != store.metadata["corpus_sha256"]:
            raise ValueError("Cleaned evidence has changed since index construction")
        vectors = np.load(index_dir / "embeddings.npy", allow_pickle=False)
        original_positions = [index for index, unit in enumerate(store.units) if not unit.is_window_copy]
        original_units = [store.units[index] for index in original_positions]
        labels, summary = cluster_texts(
            [unit.embedding_text for unit in original_units], vectors[original_positions], self.config.clustering
        )
        summary["excluded_window_units"] = len(store.units) - len(original_units)
        directory = self._new_stage("clusters")
        write_jsonl(
            directory / "assignments.jsonl", ({"id": unit.id, **row} for unit, row in zip(original_units, labels))
        )
        write_json(directory / "manifest.json", {**summary, "config": self.config.clustering.model_dump()})
        return summary

    def demand(self) -> dict:
        from ai_ready.l2.modeling import (
            build_anchors_and_routes,
            build_demand_cells,
            build_knowledge_edges,
            build_topics,
        )
        from ai_ready.l2.selection import select_anchors
        from ai_ready.l2.task_catalog import catalog_rows
        from ai_ready.retrieval.faiss_index import FaissStore

        documents = self._read_documents()
        index_dir = self.config.output_dir / "index"
        store = FaissStore.load(index_dir)
        units = store.units
        vectors = np.load(index_dir / "embeddings.npy", allow_pickle=False)
        assignments = read_jsonl(self.config.output_dir / "clusters" / "assignments.jsonl")
        cluster_summary = json.loads(
            (self.config.output_dir / "clusters" / "manifest.json").read_text(encoding="utf-8")
        )
        topics, topic_units, vector_by_id = build_topics(
            units, assignments, cluster_summary, vectors, documents, self.config.l2
        )
        anchors, routes, anchor_units = build_anchors_and_routes(
            topics, topic_units, units, documents, self.config.l2.difficulty_weights
        )
        existing_counts, existing_document_ids = self._existing_benchmark_state()
        cells = build_demand_cells(
            topics,
            anchors,
            routes,
            self.config.l2,
            existing_counts,
            existing_document_ids,
        )
        selected = select_anchors(
            anchors,
            cells,
            anchor_units,
            vector_by_id,
            self.config.l2.selection_budget,
            self.config.l2.selector,
            self.config.l2.random_state,
        )
        edges = build_knowledge_edges(documents, units, topic_units, anchors, routes, cells)
        directory = self._new_stage("demand")
        version = self._dataset_version()
        tables = {
            "topics.parquet": [topic.model_dump(mode="json") for topic in topics],
            "knowledge_anchors.parquet": [anchor.model_dump(mode="json") for anchor in anchors],
            "task_catalog.parquet": catalog_rows(),
            "task_routes.parquet": [route.model_dump(mode="json") for route in routes],
            "demand_cells.parquet": [cell.model_dump(mode="json") for cell in cells],
            "selected_anchors.parquet": selected,
            "knowledge_edges.parquet": [edge.model_dump(mode="json") for edge in edges],
        }
        for name, rows in tables.items():
            if rows:
                write_parquet(
                    self.config.output_dir / name,
                    ({"dataset_version": version, **row} for row in rows),
                    metadata={"dataset_version": version, "schema": name.removesuffix(".parquet") + "-v1"},
                    compression=self.config.storage.parquet_compression,
                )
                write_jsonl(directory / name.replace(".parquet", ".jsonl"), rows)
        summary = {
            "topics": len(topics),
            "outlier_topics": sum(topic.origin == "outlier" for topic in topics),
            "manual_topics": sum(topic.origin == "manual" for topic in topics),
            "anchors": len(anchors),
            "feasible_routes": sum(route.feasible for route in routes),
            "rejected_routes": sum(not route.feasible for route in routes),
            "valid_demand_cells": len(cells),
            "selected_anchors": len(selected),
            "knowledge_edges": len(edges),
            "selector": self.config.l2.selector,
            "existing_valid_questions": sum(existing_counts.values()),
            "existing_covered_documents": len(existing_document_ids),
        }
        write_json(directory / "manifest.json", {**summary, "config": self.config.l2.model_dump(mode="json")})
        return summary

    def publish(self) -> dict:
        from ai_ready.storage.catalog import build_duckdb, write_snapshot

        summary = {}
        if self.config.storage.create_duckdb:
            summary["duckdb"] = str(build_duckdb(self.config.output_dir))
        if self.config.storage.create_snapshot:
            config_sha = self._config_sha256()
            snapshot = write_snapshot(self.config.output_dir, self._dataset_version(), config_sha)
            summary["snapshot_files"] = len(snapshot["files"])
            summary["config_sha256"] = config_sha
        return summary

    def search(self, query: str, top_k: int) -> list[dict]:
        from ai_ready.embedding.sentence_transformer import TextEmbedder
        from ai_ready.retrieval.faiss_index import FaissStore

        if not query.strip():
            raise ValueError("Query cannot be empty")
        store = FaissStore.load(self.config.output_dir / "index")
        # Always use the index's encoder configuration, including its revision.
        embedding_config = EmbeddingConfig.model_validate(store.metadata["embedding"])
        embedding_config.device = self.config.embedding.device
        embedding_config.local_files_only = self.config.embedding.local_files_only
        embedder = TextEmbedder(embedding_config)
        return store.search(embedder.encode([query], query=True), top_k)

    def run(self) -> dict:
        if self.config.output_dir.exists() and any(self.config.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.config.output_dir}; use --output-dir")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "versions": package_versions(),
            "config": self.config.model_dump(mode="json"),
            "stages": {},
        }
        manifest_path = self.config.output_dir / "manifest.json"
        write_json(manifest_path, manifest)
        try:
            for name in ("parse", "clean", "analyze", "index", "cluster"):
                manifest["stages"][name] = getattr(self, name)()
                write_json(manifest_path, manifest)
                if name == "parse":
                    from ai_ready.storage.catalog import RunTracker

                    tracking_dir = self.config.storage.mlflow_tracking_dir or self.config.output_dir / "mlruns"
                    self._tracker = RunTracker(
                        self.config.storage.mlflow_enabled, tracking_dir, self.config.storage.mlflow_experiment
                    )
                    manifest["mlflow_run_id"] = self._tracker.start(self._dataset_version())
                if self._tracker:
                    self._tracker.log_stage(name, manifest["stages"][name])
            if self.config.l2.enabled:
                manifest["stages"]["demand"] = self.demand()
                if self._tracker:
                    self._tracker.log_stage("demand", manifest["stages"]["demand"])
            manifest["stages"]["publish"] = self.publish()
            manifest["status"] = "partial" if manifest["stages"]["parse"]["errors"] else "completed"
        except Exception as error:
            manifest.update(status="failed", error=str(error), error_type=type(error).__name__)
            raise
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)
            if self._tracker:
                self._tracker.finish(manifest["status"], manifest_path)
        return manifest

    def _read_units(self, stage: str) -> list[EvidenceUnit]:
        return [
            EvidenceUnit.model_validate(row) for row in read_jsonl(self.config.output_dir / stage / "records.jsonl")
        ]

    def _read_documents(self) -> list[DocumentRecord]:
        return [
            DocumentRecord.model_validate(row)
            for row in read_jsonl(self.config.output_dir / "parsed" / "documents.jsonl")
        ]

    def _processing_config(self) -> dict:
        config = self.config.model_dump(mode="json")
        config.pop("output_dir", None)
        config.pop("cache_dir", None)
        if config.get("storage"):
            config["storage"].pop("mlflow_tracking_dir", None)
        return config

    def _existing_benchmark_state(self) -> tuple[dict[tuple[str, str, str], int], set[str]]:
        path = self.config.l2.existing_benchmark_file
        if path is None:
            return {}, set()
        if not path.is_file():
            raise FileNotFoundError(f"Existing benchmark not found: {path}")
        if path.suffix.lower() == ".parquet":
            from ai_ready.utils.io import read_parquet

            rows = read_parquet(path)
        elif path.suffix.lower() == ".jsonl":
            rows = read_jsonl(path)
        else:
            raise ValueError("Existing benchmark must be Parquet or JSONL")
        from ai_ready.l2.task_catalog import TASK_TYPES

        counts = {}
        document_ids = set()
        for row in rows:
            if not row.get("valid", True):
                continue
            level = row.get("difficulty", row.get("target_level"))
            key = (row.get("topic_id"), row.get("task_type"), level)
            if None in key:
                raise ValueError("Existing benchmark rows require topic_id, task_type and difficulty")
            if key[1] not in TASK_TYPES:
                raise ValueError(f"Existing benchmark contains unknown task_type: {key[1]}")
            if key[2] not in {"easy", "medium", "hard"}:
                raise ValueError(f"Existing benchmark contains unknown difficulty: {key[2]}")
            counts[key] = counts.get(key, 0) + 1
            if row.get("document_id"):
                document_ids.add(row["document_id"])
            document_ids.update(value for value in row.get("document_ids", []) if value)
            for evidence in row.get("evidence", []):
                if isinstance(evidence, dict) and evidence.get("document_id"):
                    document_ids.add(evidence["document_id"])
        return counts, document_ids

    def _config_sha256(self) -> str:
        import hashlib

        payload = json.dumps(self._processing_config(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _compute_dataset_version(self, documents: list[DocumentRecord]) -> str:
        from ai_ready.storage.catalog import dataset_version

        return dataset_version([document.source_sha256 for document in documents], self._processing_config())

    def _dataset_version(self) -> str:
        manifest = json.loads((self.config.output_dir / "parsed" / "manifest.json").read_text(encoding="utf-8"))
        return manifest["dataset_version"]


def doctor() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "environment": Path(sys.prefix).name,
        "cuda_available": torch.cuda.is_available(),
        "versions": package_versions(),
    }


def print_result(value: dict | list) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))
