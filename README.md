# Value Before Full Processing: Task-Conditioned Valuation of Raw Document Batches via Trainable Proxy Datasets

This repository contains the research code for **“Value Before Full Processing: Task-Conditioned Valuation of Raw Document Batches via Trainable Proxy Datasets.”** The study asks which raw document batch is worth fully processing for a target task. CQ-Proxy estimates that value with a smaller, source-grounded trainable dataset before committing to full example generation, quality control, and training.

> This README follows manuscript revision 4. The results below are reported in that manuscript; the code release alone does not reproduce every table.

## Research question

For a fixed base model, target benchmark, training protocol, and full-processing policy, a raw batch's reference value is its downstream utility **after full processing and training**. At valuation time, the method may scan the candidate documents and use a public task-capability specification. It must not use held-out test questions, answers, or full-reference utilities.

CQ-Proxy builds a budgeted, provenance-preserving training sample for each candidate batch, runs a paired-seed proxy trial, and ranks batches by measured downstream utility. Semantic coverage, rule and exception coverage, source coverage, and quality-distribution fidelity guide **sample construction and diagnosis**; none is itself the final value estimate. The full-processing branch supplies an offline reference for evaluating the resulting decisions.

## Method

1. Freeze batch contents, the task-capability blueprint, budgets, model, training protocol, and evaluation protocol.
2. Extract source-locatable knowledge units, applicability conditions, exceptions, dependencies, and coverage obligations. Keep interdependent units in indivisible groups.
3. Select groups under a source-token budget using semantic, obligation, and source coverage together with quality-distribution fidelity. Record the selection trace.
4. Generate supervision only from selected evidence. Record each attempt as accepted, rejected with a reason, or failed; do not duplicate examples to fill a budget.
5. Train each proxy from the same base checkpoint, measure paired downstream utility on the locked benchmark, and retain seed-level results, audit records, and itemized costs.

The estimate is conditional on the specified task, model, training protocol, and full-processing policy. Changing any of them may change the batch ranking.

## Evidence reported in the paper

The controlled main comparison (Suite C) uses 16 candidate batches, 200 locked items, eight source lineages, and five paired training seeds. Under the corrected accuracy endpoint and a same-generator full-pool reference:

| Method | Spearman ρ | Regret@1 |
| --- | ---: | ---: |
| CQ-Proxy | 0.779 | 0.009 |
| Direct trial on task-relevant raw chunks | 0.548 | 0.031 |
| Direct trial on randomly selected raw chunks | 0.401 | 0.049 |

CQ-Proxy exceeds the task-relevant raw-chunk trial by 0.231 Spearman, with a 95% lineage-cluster bootstrap interval of [0.071, 0.361]. Across 200 items and five paired training seeds, full processing of the reference-best batch yields 691 correct outcomes out of 1,000; full processing of the CQ-selected batch yields 682. The resulting Regret@1 is 0.009, or 0.9 accuracy percentage points. These reference outcomes evaluate the decision after the fact; they were unavailable when CQ-Proxy selected a batch.

A bridge using an **operational full-processing reference on the same 16 candidates** gives CQ-Proxy Spearman 0.724 and Regret@1 0.012, versus 0.503 and 0.035 for task-relevant raw-chunk trials. A **separate local follow-up** with 12 batches, 46 items, and four source lineages does not preserve this advantage: CQ-Proxy has batch-level Spearman 0.302 but four-lineage equal-weight Spearman −0.714, versus 0.438 for raw random at the lineage level. The follow-up exposes failures in condition extraction, quality gating, and budget allocation, limiting claims of cross-pool stability.

Under a common accounting boundary, standardized per-batch costs are **4.18 units for CQ-Proxy, 2.79 for task-relevant raw-chunk trials, and 11.96 for full processing**. These units are not currency. CQ-Proxy incurs higher valuation cost than the strongest direct-raw comparator in exchange for lower observed decision loss in the tested pool; whether that trade-off is worthwhile depends on the cost of a wrong decision. Comparisons with stronger no-update estimators do not establish that parameter updates are necessary. The available quality-label variation also does not identify an independent contribution from the quality-fidelity term.

## Code map

| Path | Purpose |
| --- | --- |
| `ai_ready/valuation/` | Batch and provenance records, semantic mining, coverage obligations, budgeted selection, training artifacts, utility, and decision metrics |
| `ai_ready/format/`, `ai_ready/ops/`, `ai_ready/storage/` | Document parsing, cleaning, provenance, and local storage |
| `tools/value_data.py` | Command-line entry point for valuation workflows |
| `tools/acquire_cll_data.py` | Acquire the public CLL/SLL feasibility package and create local frozen files |
| `tools/build_benchmark_blueprint.py`, `tools/extract_generic_knowledge.py` | Configurable capability-blueprint and source-grounded extraction interfaces |
| `configs/valuation_cll_public_feasibility/` | Protocol, batch, and proxy-selection configurations for the CLL/SLL instance |
| `tests/` | Automated checks of core interfaces and constraints |

`ai_ready/l2/` retains an earlier benchmark-construction path for compatibility. The paper's current valuation claims center on `ai_ready/valuation/`.

## Setup and entry points

The project uses Python 3.11. On Windows, the setup script creates or checks the `aiReady` Conda environment and installs project dependencies:

```powershell
.\tools\setup_env.ps1 -Dev
conda activate aiReady
python tools/value_data.py --help
python tools/value_data.py catalog
```

Direct dependencies are declared in `pyproject.toml`; `requirements-win-py311.lock` records the Windows/Python 3.11 dependency snapshot. To acquire the public CLL/SLL feasibility package and validate the local batch design and production contract:

```powershell
python tools/acquire_cll_data.py
python tools/value_data.py validate `
  --batches configs/valuation_cll_public_feasibility/batch_suite.frozen.yaml `
  --contract configs/valuation_cll_public_feasibility/production_contract.yaml `
  --check-paths
```

This acquisition path and its configuration describe the public **12-batch CLL/SLL feasibility instance**, not the complete server reproduction package for the paper's 16-batch Suite C. A successful `validate` checks the batch design and contract structure; the current contract still contains placeholder external-tool revisions and execution commands, so it cannot directly launch the formal server experiment. Training and evaluation also require the relevant models, external tools, locked test set, and run-level records. Use `python tools/value_data.py --help` to inspect the available interfaces. Run `python -m pytest -q` for development checks.

## Release and reproducibility scope

The repository's `.gitignore` publishes source code, runnable configuration, dependency declarations, and this README. Local `data/`, `inputs/`, `outputs/`, `docs/`, and example materials are excluded. The public CLL/SLL inputs can be reacquired with the script above. The server's item-level predictions, checkpoints, cost ledger, and full-processing artifacts for the paper's main comparison are **not included** in this code release. The repository supports inspection and execution of the implementation, but **cannot by itself reproduce all reported paper results**.

Locked test items are for evaluation only. They must not influence proxy construction, prompt tuning, or example generation. A new task requires its own frozen blueprint, sources, budgets, model, training protocol, and full-processing policy, with ranking fidelity, decision loss, and cost reported separately.
