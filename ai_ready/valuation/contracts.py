"""Load, freeze and hash production contracts without depending on shell cwd."""

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ai_ready.utils.io import atomic_text, write_json
from ai_ready.valuation.adapters.catalog import ADAPTER_CATALOG
from ai_ready.valuation.schema import BatchSuite, ProductionContract, ValidationReport

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_digest(model: BaseModel) -> str:
    """Hash semantic JSON, so YAML key order and formatting cannot change identity."""

    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_digest(contract: ProductionContract) -> str:
    return canonical_digest(contract)


def suite_digest(suite: BatchSuite) -> str:
    return canonical_digest(suite)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: str | Path, model_type: type[ModelT]) -> ModelT:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return model_type.model_validate(data)


def load_batch_suite(path: str | Path) -> BatchSuite:
    return _load_yaml(path, BatchSuite)


def load_production_contract(path: str | Path) -> ProductionContract:
    return _load_yaml(path, ProductionContract)


def validate_contract_against_suite(suite: BatchSuite, contract: ProductionContract) -> ValidationReport:
    errors, warnings = [], []
    if suite.benchmark_id != contract.benchmark.benchmark_id:
        errors.append("Batch suite and production contract use different benchmark IDs")
    if suite.capability_blueprint_sha256 != contract.benchmark.capability_blueprint_sha256:
        errors.append("Batch suite and production contract use different capability blueprints")
    if len(suite.batches) < 8:
        errors.append("At least eight batches are required")
    proxy_generator = ADAPTER_CATALOG.get(contract.generation.proxy.generator)
    if proxy_generator is None:
        warnings.append(f"Proxy generator is not in the built-in catalog: {contract.generation.proxy.generator}")
    elif proxy_generator["stage"] != "generate":
        errors.append(f"Proxy generator {contract.generation.proxy.generator} is not registered for generation")
    proxy_implementation = Path(__file__).with_name("local_sft.py")
    if contract.generation.proxy.implementation_sha256 != file_sha256(proxy_implementation):
        errors.append("Proxy generation recipe does not match the checked-out local_sft.py implementation")
    unresolved_reference_fields = {
        "generation.reference.revision": contract.generation.reference.revision,
        "generation.reference.teacher_revision": contract.generation.reference.teacher_revision,
    }
    if any(
        "example" in value.lower() or "placeholder" in value.lower() for value in unresolved_reference_fields.values()
    ):
        warnings.append(
            "Fully Open Meditron reference revisions still contain placeholders; formal reference import will reject them"
        )
    for tool in contract.tools:
        catalog_entry = ADAPTER_CATALOG.get(tool.adapter)
        if catalog_entry is None:
            warnings.append(f"External adapter is not in the built-in catalog: {tool.adapter}")
        elif catalog_entry["stage"] != tool.stage and not (
            tool.adapter == "open_data_arena" and tool.stage == "evaluate"
        ):
            errors.append(f"Adapter {tool.adapter} is registered for {catalog_entry['stage']}, not {tool.stage}")
        # The in-project LLaMAFactory adapter builds one absolute argv per
        # frozen trial after data/token auditing, so a static contract argv
        # would be both incomplete and misleading for this stage.
        managed_llamafactory_train = tool.stage == "train" and tool.adapter == "llama_factory"
        if not tool.argv and not managed_llamafactory_train:
            warnings.append(f"Stage {tool.stage} has no argv binding; the plan is reviewable but not executable")
    if any(batch.content_sha256 is None for batch in suite.batches):
        warnings.append("Freeze every batch content hash before starting formal reference runs")
    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=sorted(set(warnings)),
        summary={
            "suite_id": suite.suite_id,
            "contract_id": contract.contract_id,
            "contract_sha256": contract_digest(contract),
            "batch_count": len(suite.batches),
            "seed_count": len(contract.training.seeds),
            "proxy_generator": contract.generation.proxy.generator,
            "reference_generator": contract.generation.reference.generator,
        },
    )


def freeze_contract(contract: ProductionContract, output_dir: str | Path) -> dict:
    """Write an immutable-by-content copy and a small digest manifest.

    This function does not claim that an external repository is installed.  It
    only freezes the exact protocol that every batch must subsequently use.
    """

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = contract_digest(contract)
    frozen_path = output_dir / f"{contract.contract_id}-{contract.version}-{digest[:12]}.yaml"
    if frozen_path.exists():
        existing = load_production_contract(frozen_path)
        if contract_digest(existing) != digest:
            raise FileExistsError(f"Frozen contract path has different content: {frozen_path}")
    else:
        payload = yaml.safe_dump(
            contract.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=True,
            width=120,
        )
        atomic_text(frozen_path, payload)
    manifest = {
        "schema_version": "frozen-contract-v1",
        "contract_id": contract.contract_id,
        "contract_version": contract.version,
        "contract_sha256": digest,
        "frozen_file": frozen_path.name,
        "frozen_file_sha256": file_sha256(frozen_path),
    }
    write_json(output_dir / "contract_manifest.json", manifest)
    return manifest
