"""Run a resumable, single-seed local proxy-construction screening experiment.

This is intentionally separate from the formal 252-trial experiment.  It
reuses one frozen, same-generator SFT candidate pool, varies only the raw-unit
selector, trains one LoRA adapter per batch/method, and evaluates every adapter
on the same locked 46-item benchmark.  Results are descriptive screening data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ai_ready.valuation.local_confirmation import TrainingExample
from ai_ready.valuation.local_confirmation import (
    cap_supervised_tokens as _cap_supervised_tokens,
)
from ai_ready.valuation.local_confirmation import (
    content_static_score as _content_static_score,
)
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
    select_proxy,
)

METHODS = (
    "coverage_quality",
    "coverage_only",
    "no_dependency_grouping",
    "no_source_coverage",
    "random",
    "stratified_random",
    "relevance",
    "diversity",
)
REFERENCE_METHOD = "same_generator_full_pool"
RAW_CHUNK_METHODS = ("raw_chunk_random", "raw_chunk_relevance")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _selector_config(
    method: str,
    config: dict[str, Any],
    selection_seed: int | None = None,
    token_budget: int | None = None,
) -> ProxySelectionConfig:
    selector_method = "coverage_quality" if method in {"no_dependency_grouping", "no_source_coverage"} else method
    effective_selection_seed = int(selection_seed if selection_seed is not None else config.get("selection_seed", 101))
    return ProxySelectionConfig(
        method=selector_method,
        token_budget=int(token_budget if token_budget is not None else config["selection_token_budget"]),
        random_state=effective_selection_seed if method in {"random", "stratified_random"} else 42,
        facility_weight=1.0,
        obligation_weight=1.5 if selector_method == "coverage_quality" else 1.0,
        quality_fidelity_weight=1.0,
        source_coverage_weight=0.0 if method == "no_source_coverage" else 0.5,
    )


def _selected_examples(
    batch_dir: Path,
    method: str,
    config: dict[str, Any],
    selection_seed: int | None = None,
    tokenizer=None,
) -> tuple[list[dict], dict]:
    if method in RAW_CHUNK_METHODS:
        raw_root = Path(config["_resolved_raw_chunk_sft_root"])
        raw_dir = raw_root / batch_dir.name / method
        manifest = json.loads((raw_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        sft_path = raw_dir / "accepted_sft.jsonl"
        if manifest.get("schema_version") != "raw-chunk-followup-v1" or manifest.get("batch_id") != batch_dir.name:
            raise ValueError(f"Invalid raw-chunk manifest: {raw_dir}")
        source_manifest = json.loads((batch_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("batch_content_sha256") != source_manifest.get("batch_content_sha256"):
            raise ValueError(f"Raw-chunk and CQ sources differ: {batch_dir.name}")
        if not manifest.get("target_reached"):
            raise ValueError(f"Raw-chunk generation did not reach matched training exposure: {raw_dir}")
        if _sha256(sft_path) != manifest.get("accepted_sft_sha256"):
            raise ValueError(f"Raw-chunk SFT hash mismatch: {sft_path}")
        if method == "raw_chunk_random" and manifest.get("selection_seed") != selection_seed:
            raise ValueError(f"Raw-chunk random seed mismatch: {raw_dir}")
        rows = _read_jsonl(sft_path)
        return rows, {
            "method": method,
            "raw_chunk_manifest_sha256": _sha256(raw_dir / "generation_manifest.json"),
            "source_sft_sha256": _sha256(sft_path),
            "candidate_inventory_sha256": manifest["candidate_inventory_sha256"],
            "selected_unit_count": manifest["selected_chunk_count"],
            "accepted_sample_count": len(rows),
            "used_source_tokens": manifest["used_source_tokens_estimated"],
            "selection_seed": manifest.get("selection_seed"),
        }
    accepted = _read_jsonl(batch_dir / "accepted_sft.jsonl")
    if method == REFERENCE_METHOD:
        return accepted, {
            "method": method,
            "token_budget": None,
            "used_source_tokens": None,
            "selected_unit_count": len({unit for row in accepted for unit in row["semantic_unit_ids"]}),
            "accepted_sample_count": len(accepted),
            "selection_sha256": None,
        }

    units = [ProxyUnit.model_validate(row) for row in _read_jsonl(batch_dir / "proxy_units.jsonl")]
    if method == "no_dependency_grouping":
        units = [unit.model_copy(update={"group_id": unit.unit_id}) for unit in units]
    obligations = [
        CoverageObligation.model_validate(row) for row in _read_jsonl(batch_dir / "coverage_obligations.jsonl")
    ]
    budgets = [
        int(value) for value in config.get("adaptive_selection_token_budgets", [config["selection_token_budget"]])
    ]
    if budgets != sorted(set(budgets)):
        raise ValueError("adaptive_selection_token_budgets must be unique and increasing")
    target_tokens = config.get("matched_effective_supervised_tokens")
    attempted_budgets = []
    for budget in budgets:
        attempted_budgets.append(budget)
        selection = select_proxy(units, obligations, _selector_config(method, config, selection_seed, budget))
        selected_ids = set(selection["selected_unit_ids"])
        examples = [row for row in accepted if selected_ids.intersection(row["semantic_unit_ids"])]
        if target_tokens is None or tokenizer is None:
            break
        encoded = _encode_training_examples(tokenizer, examples, int(config["max_length"]))
        if sum(row.supervised_tokens for row in encoded) >= int(target_tokens):
            break
    else:
        raise ValueError(
            f"{batch_dir.name}/{method} cannot reach {target_tokens} supervised tokens even at budget {budgets[-1]}"
        )
    selection_bytes = json.dumps(
        selection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    summary = {
        "method": method,
        "selector_method": selection["method"],
        "token_budget": selection["token_budget"],
        "used_source_tokens": selection["used_tokens"],
        "selected_unit_count": len(selected_ids),
        "accepted_sample_count": len(examples),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selection_components": selection["components"],
        "content_static_score": _content_static_score(selection["components"], config),
        "attempted_selection_token_budgets": attempted_budgets,
    }
    if method in {"random", "stratified_random"}:
        summary["selection_seed"] = int(
            selection_seed if selection_seed is not None else config.get("selection_seed", 101)
        )
    return examples, summary


def _encode_training_examples(tokenizer, rows: list[dict], max_length: int) -> list[TrainingExample]:
    encoded = []
    for row in rows:
        prompt_messages = [{"role": "user", "content": row["instruction"]}]
        full_messages = [*prompt_messages, {"role": "assistant", "content": row["response"]}]
        prompt_ids = _as_token_ids(
            tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        )
        full_ids = _as_token_ids(
            tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
        )
        full_ids = full_ids[:max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if not labels or all(label == -100 for label in labels):
            continue
        encoded.append(
            TrainingExample(
                input_ids=torch.tensor(full_ids, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.long),
                supervised_tokens=sum(label != -100 for label in labels),
            )
        )
    return encoded


def _collate_training(examples: list[TrainingExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence([row.input_ids for row in examples], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([row.labels for row in examples], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.inference_mode()
def _score_frozen_model(
    model,
    tokenizer,
    examples: list[TrainingExample],
    batch_size: int,
) -> float | None:
    """Score selected supervision before any parameter update.

    The negative response-only cross-entropy is used so larger values are
    better.  It is a frozen-model baseline, not a training result.
    """

    if not examples:
        return None
    model.eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = _collate_training(chunk, tokenizer.pad_token_id)
        batch = {name: value.to(model.device) for name, value in batch.items()}
        output = model(**batch, use_cache=False)
        chunk_supervised_tokens = sum(example.supervised_tokens for example in chunk)
        weighted_loss += float(output.loss.detach().cpu()) * chunk_supervised_tokens
        supervised_tokens += chunk_supervised_tokens
    return -(weighted_loss / supervised_tokens)


def _train_adapter(model, tokenizer, rows: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    examples = _encode_training_examples(tokenizer, rows, int(config["max_length"]))
    examples = _cap_supervised_tokens(examples, config.get("matched_effective_supervised_tokens"))
    supervised_tokens = sum(row.supervised_tokens for row in examples)
    micro_batch_size = int(config.get("training_micro_batch_size", 1))
    frozen_model_score = _score_frozen_model(model, tokenizer, examples, micro_batch_size)
    if not examples:
        return {
            "trained": False,
            "encoded_sample_count": 0,
            "effective_supervised_tokens": 0,
            "optimizer_steps": 0,
            "mean_training_loss": None,
            "frozen_model_score": None,
            "wall_seconds": 0.0,
        }

    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    adapter_model = get_peft_model(model, lora_config)
    adapter_model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter_model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=0.0,
    )
    losses = []
    optimizer_steps = 0
    started = time.perf_counter()
    fixed_optimizer_steps = config.get("fixed_optimizer_steps")
    if fixed_optimizer_steps is not None:
        batch = _collate_training(examples, tokenizer.pad_token_id)
        batch = {name: value.to(adapter_model.device) for name, value in batch.items()}
        for _ in range(int(fixed_optimizer_steps)):
            optimizer.zero_grad(set_to_none=True)
            output = adapter_model(**batch, use_cache=False)
            loss = output.loss
            losses.append(float(loss.detach().cpu()))
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    else:
        accumulation = int(config["gradient_accumulation_steps"])
        for epoch in range(int(config["num_train_epochs"])):
            order = list(range(len(examples)))
            random.Random(int(config["training_seed"]) + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            chunks = [order[start : start + micro_batch_size] for start in range(0, len(order), micro_batch_size)]
            for step, indices in enumerate(chunks, start=1):
                batch = _collate_training(
                    [examples[index] for index in indices],
                    tokenizer.pad_token_id,
                )
                batch = {name: value.to(adapter_model.device) for name, value in batch.items()}
                output = adapter_model(**batch, use_cache=False)
                loss = output.loss
                losses.append(float(loss.detach().cpu()))
                (loss / accumulation).backward()
                if step % accumulation == 0 or step == len(chunks):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
    wall_seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    for parameter in adapter_model.parameters():
        parameter.grad = None
    del optimizer
    if "batch" in locals():
        del batch
    if "output" in locals():
        del output
    if "loss" in locals():
        del loss
    gc.collect()
    torch.cuda.empty_cache()
    adapter_model.eval()
    return {
        "model": adapter_model,
        "summary": {
            "trained": True,
            "encoded_sample_count": len(examples),
            "effective_supervised_tokens": supervised_tokens,
            "optimizer_steps": optimizer_steps,
            "mean_training_loss": float(sum(losses) / len(losses)),
            "frozen_model_score": frozen_model_score,
            "wall_seconds": wall_seconds,
        },
    }


def _evaluation_records(tokenizer, benchmark_rows: list[dict], max_length: int) -> list[dict[str, Any]]:
    records = []
    for item_index, row in enumerate(benchmark_rows):
        option_lines = "\n".join(f"{letter}. {text}" for letter, text in zip(("A", "B", "C", "D"), row["options"]))
        messages = [
            {
                "role": "system",
                "content": "Choose the single best answer to the medical multiple-choice question. Reply with only A, B, C, or D.",
            },
            {"role": "user", "content": f"{row['question']}\n\n{option_lines}"},
        ]
        prompt_ids = _as_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if len(prompt_ids) >= max_length - 2:
            prompt_ids = prompt_ids[-(max_length - 2) :]
        for option_index, letter in enumerate(("A", "B", "C", "D")):
            option_ids = tokenizer.encode(letter, add_special_tokens=False)
            records.append(
                {
                    "item_index": item_index,
                    "benchmark_item_id": row["benchmark_item_id"],
                    "answer_index": int(row["answer_index"]),
                    "option_index": option_index,
                    "input_ids": torch.tensor([*prompt_ids, *option_ids], dtype=torch.long),
                    "prompt_length": len(prompt_ids),
                    "option_length": len(option_ids),
                }
            )
    return records


@torch.inference_mode()
def _evaluate(model, tokenizer, records: list[dict[str, Any]], batch_size: int) -> tuple[dict, list[dict]]:
    model.eval()
    scores: dict[tuple[int, int], float] = {}
    single_token_options = all(row["option_length"] == 1 for row in records)
    if single_token_options:
        item_records = sorted(
            (records[index] for index in range(0, len(records), 4)),
            key=lambda row: row["prompt_length"],
        )
        for start in range(0, len(item_records), batch_size):
            chunk = item_records[start : start + batch_size]
            prompt_ids = [row["input_ids"][: row["prompt_length"]] for row in chunk]
            max_length = max(len(value) for value in prompt_ids)
            input_ids = torch.full(
                (len(chunk), max_length),
                tokenizer.pad_token_id,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, value in enumerate(prompt_ids):
                input_ids[row_index, -len(value) :] = value
                attention_mask[row_index, -len(value) :] = 1
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            output = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                position_ids=position_ids.to(model.device),
                logits_to_keep=1,
                use_cache=False,
            )
            next_token_log_probs = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1).cpu()
            for row_index, first in enumerate(chunk):
                base_index = first["item_index"] * 4
                for option_index in range(4):
                    option = records[base_index + option_index]
                    token_id = int(option["input_ids"][-1])
                    scores[(first["item_index"], option_index)] = float(next_token_log_probs[row_index, token_id])
    else:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            input_ids = pad_sequence(
                [row["input_ids"] for row in chunk],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            output = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                use_cache=False,
            )
            log_probs = torch.log_softmax(output.logits.float(), dim=-1).cpu()
            for row_index, row in enumerate(chunk):
                score = 0.0
                for offset in range(row["option_length"]):
                    token_position = row["prompt_length"] + offset
                    token_id = int(row["input_ids"][token_position])
                    score += float(log_probs[row_index, token_position - 1, token_id])
                scores[(row["item_index"], row["option_index"])] = score / row["option_length"]

    predictions = []
    correct = 0
    item_count = len(records) // 4
    for item_index in range(item_count):
        option_scores = [scores[(item_index, option_index)] for option_index in range(4)]
        predicted = max(range(4), key=lambda option_index: (option_scores[option_index], -option_index))
        first = records[item_index * 4]
        correct += int(predicted == first["answer_index"])
        predictions.append(
            {
                "benchmark_item_id": first["benchmark_item_id"],
                "answer_index": first["answer_index"],
                "predicted_index": predicted,
                "option_log_likelihoods": option_scores,
                "correct": predicted == first["answer_index"],
            }
        )
    return {"correct_item_count": correct, "item_count": item_count, "accuracy": correct / item_count}, predictions


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "local-proxy-screening-v1":
        raise ValueError("Unsupported local screening configuration")
    return value


def _load_completed(path: Path) -> set[tuple[str, str, int, int | None]]:
    if not path.is_file():
        return set()
    return {
        (row["batch_id"], row["method"], int(row["training_seed"]), row.get("selection_seed"))
        for row in _read_jsonl(path)
        if row.get("status") == "completed"
    }


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the local 7B screening experiment")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    training_seed = int(args.training_seed if args.training_seed is not None else config["training_seed"])
    selection_seed = int(args.selection_seed if args.selection_seed is not None else config["selection_seed"])
    project_root = config_path.parents[2]
    source_root = (project_root / config["source_sft_root"]).resolve()
    if config.get("raw_chunk_sft_root"):
        config["_resolved_raw_chunk_sft_root"] = str((project_root / config["raw_chunk_sft_root"]).resolve())
    benchmark_path = (project_root / config["benchmark_path"]).resolve()
    model_path = Path(config["model_path"]).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trial_results.jsonl"
    predictions_path = output_dir / "item_predictions.jsonl"
    completed = _load_completed(trials_path)

    methods = list(config["methods"])
    if config.get("include_full_pool_reference"):
        methods.append(REFERENCE_METHOD)
    if args.methods:
        unknown = set(args.methods) - set(methods)
        if unknown:
            raise ValueError(f"Methods are not enabled by the config: {sorted(unknown)}")
        methods = args.methods
    batch_dirs = sorted(path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("B"))
    if args.batches:
        batch_dirs = [path for path in batch_dirs if path.name in set(args.batches)]
    if not batch_dirs:
        raise ValueError("No batch directories selected")

    benchmark_rows = _read_jsonl(benchmark_path)
    if len(benchmark_rows) != 46:
        raise ValueError(f"The locked screening benchmark must contain 46 items, got {len(benchmark_rows)}")
    _set_seed(training_seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=bool(config["quantization"]["use_double_quant"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        device_map="cuda",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    evaluation_records = _evaluation_records(tokenizer, benchmark_rows, int(config["max_length"]))
    base_metrics, base_predictions = _evaluate(
        model, tokenizer, evaluation_records, int(config["evaluation_batch_size"])
    )
    (output_dir / "base_model_result.json").write_text(
        json.dumps(
            {"schema_version": "local-screening-base-result-v1", **base_metrics, "predictions": base_predictions},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    run_manifest = {
        "schema_version": "local-proxy-screening-run-v1",
        "artifact_role": config["artifact_role"],
        "config_sha256": _sha256(config_path),
        "model_name": config["model_name"],
        "model_path": str(model_path),
        "model_revision": config["model_revision"],
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "training_seed": training_seed,
        "selection_seed": selection_seed,
        "batch_ids": [path.name for path in batch_dirs],
        "methods": methods,
        "expected_trial_count": len(batch_dirs) * len(methods),
        "base_model_accuracy": base_metrics["accuracy"],
        "base_model_correct_item_count": base_metrics["correct_item_count"],
        "benchmark_item_count": base_metrics["item_count"],
        "result_files": [str(trials_path), str(predictions_path)],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    executed = 0
    for batch_dir in batch_dirs:
        for method in methods:
            trial_selection_seed = (
                selection_seed if method in {"random", "stratified_random", "raw_chunk_random"} else None
            )
            key = (batch_dir.name, method, training_seed, trial_selection_seed)
            if key in completed:
                continue
            if args.max_trials is not None and executed >= args.max_trials:
                return 0
            _set_seed(training_seed)
            rows, selection_summary = _selected_examples(batch_dir, method, config, selection_seed, tokenizer=tokenizer)
            trial_config = {**config, "training_seed": training_seed}
            if method == REFERENCE_METHOD:
                trial_config.pop("matched_effective_supervised_tokens", None)
                trial_config.pop("fixed_optimizer_steps", None)
            training = _train_adapter(model, tokenizer, rows, trial_config)
            if "model" in training:
                adapter_model = training["model"]
                training_summary = training["summary"]
                metrics, predictions = _evaluate(
                    adapter_model, tokenizer, evaluation_records, int(config["evaluation_batch_size"])
                )
                model = adapter_model.unload()
                if hasattr(model, "peft_config"):
                    delattr(model, "peft_config")
                if hasattr(model, "_hf_peft_config_loaded"):
                    model._hf_peft_config_loaded = False
                del adapter_model
            else:
                training_summary = training
                metrics, predictions = base_metrics, base_predictions
            trial_id = hashlib.sha256(
                f"{batch_dir.name}|{method}|{training_seed}|{trial_selection_seed}|{_sha256(config_path)}".encode()
            ).hexdigest()[:24]
            for prediction in predictions:
                _append_jsonl(
                    predictions_path,
                    {"trial_id": trial_id, "batch_id": batch_dir.name, "method": method, **prediction},
                )
            row = {
                "schema_version": "local-proxy-screening-trial-v1",
                "status": "completed",
                "trial_id": trial_id,
                "batch_id": batch_dir.name,
                "method": method,
                "training_seed": training_seed,
                "selection_seed": trial_selection_seed,
                "selection": selection_summary,
                "training": training_summary,
                "benchmark": metrics,
                "base_model_accuracy": base_metrics["accuracy"],
                "utility": metrics["accuracy"] - base_metrics["accuracy"],
                "completed_at_unix": time.time(),
            }
            _append_jsonl(trials_path, row)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            executed += 1
            gc.collect()
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=[*METHODS, *RAW_CHUNK_METHODS, REFERENCE_METHOD])
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--selection-seed", type=int)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
