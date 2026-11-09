"""Run existing DJ operators and retain a complete evidence audit."""

from collections import defaultdict

import yaml

from ai_ready.config import CleaningConfig
from ai_ready.core.schema import EvidenceUnit


def clean_units(units: list[EvidenceUnit], config: CleaningConfig) -> tuple[list[EvidenceUnit], list[dict]]:
    from data_juicer.core.data import NestedDataset
    from data_juicer.ops import OPERATORS, Filter, Mapper
    from data_juicer.ops.deduplicator.document_deduplicator import DocumentDeduplicator
    from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
        DocumentMinhashDeduplicator,
    )
    from data_juicer.utils.constant import HashKeys
    from datasets import disable_caching

    if not units:
        return [], []
    # In-memory inputs and stateful deduplicators must not reuse stale filter masks.
    disable_caching()
    by_id = {unit.id: unit.model_copy(deep=True) for unit in units}
    if len(by_id) != len(units):
        raise ValueError("Evidence IDs must be unique")
    dataset = NestedDataset.from_list([{"id": unit.id, "text": unit.text} for unit in units])
    recipe = yaml.safe_load(config.recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or not isinstance(recipe.get("process"), list):
        raise ValueError("Data-Juicer recipe requires a process list")
    audit = []
    for step in recipe["process"]:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError("Each recipe step must contain exactly one Data-Juicer operator")
        name, params = next(iter(step.items()))
        op_class = OPERATORS.modules.get(name)
        if op_class is None or not issubclass(op_class, (Mapper, Filter)):
            raise ValueError(f"Use a built-in mapper/filter in the cleaning recipe: {name}")
        op = op_class(**(params or {}), text_key="text", num_proc=1, auto_op_parallelism=False)
        previous = set(dataset["id"])
        dataset = op.run(dataset)
        current = set(dataset["id"])
        if not current.issubset(previous) or len(current) != len(dataset):
            raise ValueError("L1 cleaning operators must preserve evidence IDs and cannot expand rows")
        audit.extend({"id": item, "action": "filtered", "operator": name} for item in sorted(previous - current))
        if not len(dataset):
            return [], audit
    for row in dataset:
        by_id[row["id"]].text = row["text"]
    # Version identity is part of the dedup key. Equal text in two versions is
    # retained; document-level relations describe their relationship.
    minimal_rows = [
        {
            "id": row["id"],
            "text": f"{by_id[row['id']].document_version_id}\0{by_id[row['id']].dedup_text}",
        }
        for row in dataset
    ]
    dataset = NestedDataset.from_list(minimal_rows)
    exact = DocumentDeduplicator(
        lowercase=False, ignore_non_character=False, text_key="text", num_proc=1, auto_op_parallelism=False
    )
    # Hashing and duplicate selection are both upstream DJ operations.
    hashed = dataset.map(exact.compute_hash, load_from_cache_file=False)
    groups = defaultdict(list)
    for row in hashed:
        groups[row[HashKeys.hash]].append(row["id"])
    deduplicated, _ = exact.process(hashed)
    cleaned = []
    for row in deduplicated:
        unit = by_id[row["id"]]
        for duplicate_id in groups[row[HashKeys.hash]]:
            if duplicate_id == unit.id:
                continue
            duplicate = by_id[duplicate_id]
            unit.sources.extend(duplicate.sources)
            unit.duplicate_ids.extend([duplicate_id, *duplicate.duplicate_ids])
            audit.append(
                {"id": duplicate_id, "action": "exact_duplicate", "canonical_id": unit.id, "operator": exact._name}
            )
        cleaned.append(unit)
    if config.near_duplicates and len(cleaned) > 1:
        # Very short samples have empty shingles and must not all collapse together.
        eligible = [unit for unit in cleaned if len(unit.dedup_text) >= config.minhash_window]
        if len(eligible) > 1:
            near = DocumentMinhashDeduplicator(
                tokenization="character",
                window_size=config.minhash_window,
                lowercase=False,
                num_permutations=config.minhash_permutations,
                jaccard_threshold=config.minhash_threshold,
                text_key="text",
                num_proc=1,
                auto_op_parallelism=False,
            )
            candidates = NestedDataset.from_list([{"id": unit.id, "text": unit.dedup_text} for unit in eligible])
            retained = set(near.run(candidates)["id"])
            for unit in eligible:
                if unit.id not in retained:
                    unit.near_duplicate_candidate = True
                    audit.append({"id": unit.id, "action": "near_duplicate_flagged", "operator": near._name})
    return cleaned, audit
