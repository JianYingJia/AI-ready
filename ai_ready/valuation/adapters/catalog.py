"""Known external tools and the artifact boundary expected by this project."""

from copy import deepcopy

ADAPTER_CATALOG = {
    "coverage_guided_sft": {
        "stage": "generate",
        "repository": "local://ai-ready-benchmark/ai_ready/valuation/local_sft.py",
        "input": ["frozen raw batch", "capability blueprint", "proxy selection config", "production contract"],
        "output": [
            "accepted_sft.jsonl",
            "selection_manifest.json",
            "proxy_units.jsonl",
            "coverage_obligations.jsonl",
            "dispositions.jsonl",
            "generation_manifest.json",
        ],
    },
    "data_juicer": {
        "stage": "preprocess",
        "repository": "https://github.com/datajuicer/data-juicer",
        "input": ["raw documents", "source registry"],
        "output": ["cleaned_docs.jsonl", "preprocess_manifest.json"],
    },
    "fully_open_meditron": {
        "stage": "generate",
        "repository": "https://github.com/EPFLiGHT/FullyOpenMeditron",
        "input": ["complete frozen batch", "eligible_units.jsonl", "production contract"],
        "output": ["normalized_fom.jsonl", "fully_open_meditron_run_manifest.json"],
    },
    "bonito": {
        "stage": "generate",
        "repository": "https://github.com/BatsResearch/bonito",
        "input": ["cleaned_docs.jsonl"],
        "output": ["candidate_sft.jsonl", "generation_manifest.json"],
    },
    "open_data_arena": {
        "stage": "train",
        "repository": "https://github.com/OpenDataArena/OpenDataArena-Tool",
        "input": ["accepted_sft.jsonl", "training_config"],
        "output": ["checkpoint_manifest.json", "training_metrics.json"],
    },
    "llama_factory": {
        "stage": "train",
        "repository": "https://github.com/hiyouga/LlamaFactory",
        "input": ["accepted_sft.jsonl", "training_config"],
        "output": ["checkpoint_manifest.json", "training_metrics.json"],
    },
    "evalscope": {
        "stage": "evaluate",
        "repository": "https://github.com/modelscope/evalscope",
        "input": ["checkpoint", "locked benchmark manifest"],
        "output": ["predictions.jsonl", "metrics.json"],
    },
    "opencompass": {
        "stage": "evaluate",
        "repository": "https://github.com/open-compass/opencompass",
        "input": ["checkpoint", "locked benchmark manifest"],
        "output": ["predictions.jsonl", "metrics.json"],
    },
}


def adapter_catalog() -> dict:
    """Return a copy so callers cannot mutate the process-wide registry."""

    return deepcopy(ADAPTER_CATALOG)
