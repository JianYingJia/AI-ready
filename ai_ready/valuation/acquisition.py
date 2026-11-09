"""Acquire and freeze the public CLL feasibility data package.

The public package is intentionally separate from a user's locked benchmark.
It provides real, licensed documents and a reproducible proxy benchmark for
pipeline smoke tests; it is not evidence that the selector predicts value on
an unseen customer benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import requests
import yaml
from bs4 import BeautifulSoup, Tag

from ai_ready.valuation.batches import freeze_batch_suite

SNAPSHOT_DATE = "2026-09-13"
MEDMCQA_COMMIT = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
MEDMCQA_GITHUB_COMMIT = "c59ef14ca1990266c4107c7864b45a20fd93e5e0"
MEDMCQA_TRAIN_SHA256 = "b119434ba551517a6ec0ba1f7e0b4c029165ed284a4704f262ce37c791c493c5"
USER_AGENT = "ai-ready-benchmark/0.1 (public research data acquisition)"


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    title: str
    canonical_url: str
    download_urls: tuple[str, ...]
    raw_filename: str
    source_format: str
    version: str
    language: str
    license_id: str
    license_url: str
    allowed_uses: tuple[str, ...]
    knowledge_cutoff: str


PUBLIC_SOURCES = (
    SourceDefinition(
        source_id="nci_cll_professional",
        title="Chronic Lymphocytic Leukemia Treatment (PDQ) - Health Professional Version",
        canonical_url="https://www.cancer.gov/types/leukemia/hp/cll-treatment-pdq",
        download_urls=("https://www.cancer.gov/types/leukemia/hp/cll-treatment-pdq",),
        raw_filename="nci_cll_professional.html",
        source_format="nci_html",
        version="updated-2025-04-25_snapshot-2026-09-13",
        language="en",
        license_id="US-government-text-reuse",
        license_url="https://www.cancer.gov/policies/copyright-reuse",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2025-04-25",
    ),
    SourceDefinition(
        source_id="nci_cll_patient",
        title="Chronic Lymphocytic Leukemia Treatment (PDQ) - Patient Version",
        canonical_url="https://www.cancer.gov/types/leukemia/patient/cll-treatment-pdq",
        download_urls=("https://www.cancer.gov/types/leukemia/patient/cll-treatment-pdq",),
        raw_filename="nci_cll_patient.html",
        source_format="nci_html",
        version="updated-2024-10-15_snapshot-2026-09-13",
        language="en",
        license_id="US-government-text-reuse",
        license_url="https://www.cancer.gov/policies/copyright-reuse",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2024-10-15",
    ),
    SourceDefinition(
        source_id="nci_aml_professional",
        title="Acute Myeloid Leukemia Treatment (PDQ) - Health Professional Version",
        canonical_url="https://www.cancer.gov/types/leukemia/hp/adult-aml-treatment-pdq",
        download_urls=("https://www.cancer.gov/types/leukemia/hp/adult-aml-treatment-pdq",),
        raw_filename="nci_aml_professional.html",
        source_format="nci_html",
        version="updated-2025-03-14_snapshot-2026-09-13",
        language="en",
        license_id="US-government-text-reuse",
        license_url="https://www.cancer.gov/policies/copyright-reuse",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2025-03-14",
    ),
    SourceDefinition(
        source_id="nci_cml_professional",
        title="Chronic Myeloid Leukemia Treatment (PDQ) - Health Professional Version",
        canonical_url="https://www.cancer.gov/types/leukemia/hp/cml-treatment-pdq",
        download_urls=("https://www.cancer.gov/types/leukemia/hp/cml-treatment-pdq",),
        raw_filename="nci_cml_professional.html",
        source_format="nci_html",
        version="updated-2025-03-13_snapshot-2026-09-13",
        language="en",
        license_id="US-government-text-reuse",
        license_url="https://www.cancer.gov/policies/copyright-reuse",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2025-03-13",
    ),
    SourceDefinition(
        source_id="pmc11171037",
        title="Treatment Sequencing in Chronic Lymphocytic Leukemia in 2024",
        canonical_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11171037/",
        download_urls=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11171037/fullTextXML",
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC11171037/unicode",
        ),
        raw_filename="pmc11171037.xml",
        source_format="jats_xml",
        version="PMCID-PMC11171037_2024",
        language="en",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2024-06-01",
    ),
    SourceDefinition(
        source_id="pmc11505876",
        title="Recent Advances in the Molecular Biology of Chronic Lymphocytic Leukemia",
        canonical_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11505876/",
        download_urls=(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11505876/fullTextXML",
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC11505876/unicode",
        ),
        raw_filename="pmc11505876.xml",
        source_format="jats_xml",
        version="PMCID-PMC11505876_2024",
        language="en",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        allowed_uses=("research", "model-training"),
        knowledge_cutoff="2024-10-14",
    ),
)

CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "capability_id": "diagnosis_and_differential",
        "description": "Recognize CLL/SLL and distinguish it from related leukemias and lymphomas.",
        "weight": 1.0,
        "keywords": (
            "diagnos",
            "differential",
            "monoclonal b-cell",
            "mbl",
            "mantle cell",
            "clonal b",
            "small lymphocytic",
        ),
    },
    {
        "capability_id": "pathology_and_immunophenotype",
        "description": "Apply morphology, lineage, immunophenotype, and laboratory features of CLL/SLL.",
        "weight": 1.0,
        "keywords": (
            "immunophenotype",
            "cd5",
            "cd19",
            "cd20",
            "cd23",
            "smudge",
            "lymphocyte",
            "flow cytometry",
            "morpholog",
        ),
    },
    {
        "capability_id": "complications",
        "description": "Identify immune, infectious, metabolic, and secondary-malignancy complications.",
        "weight": 1.0,
        "keywords": (
            "complication",
            "infection",
            "autoimmune",
            "hemolytic",
            "thrombocytopen",
            "hypogammaglobulin",
            "tumor lysis",
            "secondary malign",
        ),
    },
    {
        "capability_id": "prognosis_and_transformation",
        "description": "Interpret prognostic factors and recognize Richter transformation.",
        "weight": 1.0,
        "keywords": (
            "prognos",
            "survival",
            "richter",
            "transform",
            "del(17p",
            "tp53",
            "ighv",
            "risk",
        ),
    },
    {
        "capability_id": "treatment_indication",
        "description": "Decide when observation or initiation of treatment is appropriate.",
        "weight": 1.0,
        "keywords": (
            "asymptomatic",
            "watchful",
            "observation",
            "indication for treatment",
            "initiation of treatment",
            "withheld",
            "active disease",
        ),
    },
    {
        "capability_id": "treatment_selection",
        "description": "Select and sequence CLL/SLL therapies under disease and patient constraints.",
        "weight": 1.0,
        "keywords": (
            "treatment",
            "therapy",
            "frontline",
            "first-line",
            "relapsed",
            "refractory",
            "venetoclax",
            "ibrutinib",
            "acalabrutinib",
            "zanubrutinib",
            "chlorambucil",
            "fludarabine",
        ),
    },
)

EXCLUDED_NCI_SECTIONS = (
    "references",
    "latest updates",
    "about this pdq",
    "permission to use",
    "disclaimer",
    "contact us",
)

CLL_PATTERN = re.compile(
    r"\b(?:CLL|SLL)\b|(?i:chronic\s+lympho(?:cytic|id)\s+leu?k(?:a)?emia|small\s+lymphocytic\s+lymphoma)"
)
TEMPORAL_RISK_PATTERN = re.compile(
    r"(?i)\b(?:preferred agent|most common chemotherapeutic|first[- ]line|chlorambucil|fludarabine|FCR|BR)\b"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _download(urls: Sequence[str], destination: Path, *, refresh: bool, expected_sha256: str | None = None) -> str:
    """Download atomically and return the URL that supplied the bytes.

    Existing snapshots are reused by default so a moving web page cannot
    silently mutate a registered experiment. ``--refresh`` is an explicit new
    acquisition and the resulting hashes must be reviewed before use.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not refresh:
        observed = file_sha256(destination)
        if expected_sha256 and observed != expected_sha256:
            raise ValueError(f"Cached file hash mismatch for {destination}: {observed}")
        return "local-cache"

    errors: list[str] = []
    temporary = destination.with_suffix(destination.suffix + ".part")
    for url in urls:
        try:
            with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            observed = file_sha256(temporary)
            if expected_sha256 and observed != expected_sha256:
                raise ValueError(f"Downloaded hash mismatch: expected {expected_sha256}, observed {observed}")
            temporary.replace(destination)
            return url
        except Exception as error:  # pragma: no cover - exercised only when an upstream is unavailable
            temporary.unlink(missing_ok=True)
            errors.append(f"{url}: {type(error).__name__}: {error}")
    raise RuntimeError("All download locations failed:\n" + "\n".join(errors))


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    return value.strip()


def _table_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if cells and any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in rows[1:]),
        ]
    )


def extract_nci_blocks(payload: bytes) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(payload, "html.parser")
    article = soup.find("main", id="main-content") or soup.find("article")
    if article is None:
        raise ValueError("NCI page does not contain the expected main article")
    for unwanted in article.find_all(["script", "style", "nav", "aside", "figure", "img", "svg", "form"]):
        unwanted.decompose()

    title_tag = article.find("h1") or soup.find("h1")
    title = _normalize_text(title_tag.get_text(" ", strip=True)) if title_tag else "NCI cancer information"
    headings: list[str] = []
    blocks: list[dict[str, str]] = []
    previous: tuple[str, str] | None = None
    for element in article.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        if element.name and element.name.startswith("h"):
            level = int(element.name[1])
            heading = _normalize_text(element.get_text(" ", strip=True))
            headings = headings[: max(level - 1, 0)]
            while len(headings) < level - 1:
                headings.append("")
            headings.append(heading)
            continue
        # Paragraphs inside list items or table cells are emitted by their
        # containing structure; skipping them prevents accidental duplication.
        if element.name == "p" and element.find_parent(["li", "td", "th"]):
            continue
        if element.name == "li" and element.find_parent("li"):
            continue
        if element.name != "table" and element.find_parent("table"):
            continue
        section = " > ".join(part for part in headings if part)
        if any(marker in section.casefold() for marker in EXCLUDED_NCI_SECTIONS):
            continue
        if element.name == "table":
            text = _table_markdown(element)
            kind = "table"
        else:
            text = _normalize_text(element.get_text(" ", strip=True))
            kind = "list_item" if element.name == "li" else "paragraph"
        if len(text) < 20 or text.casefold().startswith("enlarge image"):
            continue
        identity = (section, text)
        if identity == previous:
            continue
        blocks.append({"kind": kind, "section": section, "text": text})
        previous = identity
    if len(blocks) < 20:
        raise ValueError(f"NCI extraction produced too few blocks: {len(blocks)}")
    return title, blocks


def _jats_section_path(element: Tag) -> str:
    sections = []
    for parent in reversed(list(element.parents)):
        if isinstance(parent, Tag) and parent.name == "sec":
            title = parent.find("title", recursive=False)
            if title:
                sections.append(_normalize_text(title.get_text(" ", strip=True)))
    return " > ".join(sections)


def extract_jats_blocks(payload: bytes) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(payload, "xml")
    title_tag = soup.find("article-title")
    body = soup.find("body")
    if title_tag is None or body is None:
        raise ValueError("PMC payload is not JATS full text")
    title = _normalize_text(title_tag.get_text(" ", strip=True))
    blocks: list[dict[str, str]] = []

    abstract = soup.find("abstract")
    if abstract:
        for paragraph in abstract.find_all("p"):
            text = _normalize_text(paragraph.get_text(" ", strip=True))
            if len(text) >= 20:
                blocks.append({"kind": "paragraph", "section": "Abstract", "text": text})

    for element in body.find_all(["p", "table-wrap"]):
        if element.name == "p" and element.find_parent(["table-wrap", "fig", "ref-list", "fn-group"]):
            continue
        if element.name == "table-wrap" and element.find_parent("table-wrap"):
            continue
        section = _jats_section_path(element)
        if element.name == "table-wrap":
            table = element.find("table")
            text = _table_markdown(table) if table else ""
            kind = "table"
        else:
            text = _normalize_text(element.get_text(" ", strip=True))
            kind = "paragraph"
        if len(text) >= 20:
            blocks.append({"kind": kind, "section": section, "text": text})
    if len(blocks) < 20:
        raise ValueError(f"PMC extraction produced too few blocks: {len(blocks)}")
    return title, blocks


def _render_document(source: SourceDefinition, title: str, blocks: Sequence[dict[str, str]], raw_sha256: str) -> str:
    lines = [
        "---",
        f"source_id: {source.source_id}",
        f"source_title: {json.dumps(source.title, ensure_ascii=False)}",
        f"source_url: {source.canonical_url}",
        f"source_version: {source.version}",
        f"license: {source.license_id}",
        f"raw_sha256: {raw_sha256}",
        "adaptation: Text-only extraction; figures, site navigation, and reference-only sections removed.",
        "---",
        "",
        f"# {title}",
        "",
    ]
    previous_section = None
    for block in blocks:
        if block["section"] and block["section"] != previous_section:
            lines.extend([f"## {block['section']}", ""])
            previous_section = block["section"]
        if block["kind"] == "list_item":
            lines.append(f"- {block['text']}")
        else:
            lines.append(block["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _keyword_score(block: dict[str, str], keywords: Sequence[str]) -> int:
    heading = block["section"].casefold()
    text = block["text"].casefold()
    return sum(3 for keyword in keywords if keyword.casefold() in heading) + sum(
        1 for keyword in keywords if keyword.casefold() in text
    )


def _build_relevant_units(source: SourceDefinition, blocks: Sequence[dict[str, str]]) -> dict[str, str]:
    units: dict[str, str] = {}
    used: set[int] = set()
    # Scarcer capability evidence is allocated first. This keeps an individual
    # source paragraph in one low-redundancy unit whenever possible.
    ranked: dict[str, list[tuple[int, int]]] = {}
    for capability in CAPABILITIES:
        candidates = [
            (_keyword_score(block, capability["keywords"]), index)
            for index, block in enumerate(blocks)
            if 60 <= len(block["text"]) <= 6000
        ]
        ranked[capability["capability_id"]] = sorted(
            (item for item in candidates if item[0] > 0), key=lambda item: (-item[0], item[1])
        )

    order = sorted(CAPABILITIES, key=lambda item: len(ranked[item["capability_id"]]))
    for capability in order:
        capability_id = capability["capability_id"]
        chosen: list[int] = []
        total = 0
        for _, index in ranked[capability_id]:
            if index in used:
                continue
            chosen.append(index)
            used.add(index)
            total += len(blocks[index]["text"])
            if len(chosen) >= 8 or total >= 6000:
                break
        if not chosen:
            raise ValueError(f"No source-grounded blocks found for capability {capability_id}")
        lines = [
            f"# CLL evidence unit: {capability_id}",
            "",
            f"Source: {source.title}",
            f"Source URL: {source.canonical_url}",
            f"Source version: {source.version}",
            "",
        ]
        previous_section = None
        for index in sorted(chosen):
            block = blocks[index]
            if block["section"] and block["section"] != previous_section:
                lines.extend([f"## {block['section']}", ""])
                previous_section = block["section"]
            lines.extend([block["text"], ""])
        units[capability_id] = "\n".join(lines).rstrip() + "\n"
    return units


def _build_irrelevant_units(
    sources_and_blocks: Sequence[tuple[SourceDefinition, Sequence[dict[str, str]]]], target_sizes: Sequence[int]
) -> dict[str, str]:
    pool: list[tuple[SourceDefinition, dict[str, str]]] = []
    seen: set[str] = set()
    for source, blocks in sources_and_blocks:
        for block in blocks:
            normalized = _normalize_for_matching(block["text"])
            if len(block["text"]) < 80 or normalized in seen:
                continue
            seen.add(normalized)
            pool.append((source, block))
    if not pool:
        raise ValueError("No AML/CML blocks available for low-relevance controls")

    units: dict[str, str] = {}
    for index, target_size in enumerate(target_sizes, start=1):
        chosen: list[tuple[SourceDefinition, dict[str, str]]] = []
        total = 0
        # Pick whole paragraphs that bring the control unit closest to the
        # corresponding CLL unit. This controls volume without truncating a
        # medical condition, exception, or table row.
        target_body_size = max(target_size - 500, 500)
        while pool and (total < target_body_size or not chosen):
            best_index = min(
                range(len(pool)),
                key=lambda candidate_index: abs(target_body_size - total - len(pool[candidate_index][1]["text"])),
            )
            source, block = pool.pop(best_index)
            chosen.append((source, block))
            total += len(block["text"])
            if total >= target_body_size * 0.95:
                break
        if not chosen:
            raise ValueError("AML/CML pool is too small to size-match controlled units")
        lines = [
            f"# Non-CLL leukemia control unit {index:02d}",
            "",
            "This unit is medically relevant to leukemia but outside the target CLL/SLL capability blueprint.",
            "",
        ]
        previous_source = None
        for source, block in chosen:
            if source.source_id != previous_source:
                lines.extend([f"Source: {source.title}", f"Source URL: {source.canonical_url}", ""])
                previous_source = source.source_id
            if block["section"]:
                lines.extend([f"## {block['section']}", ""])
            lines.extend([block["text"], ""])
        units[f"control_{index:02d}"] = "\n".join(lines).rstrip() + "\n"
    return units


def _reset_flat_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            raise ValueError(f"Refusing to remove unexpected nested directory: {child}")
        child.unlink()


def _write_batch(
    batch_root: Path,
    batch_id: str,
    documents: dict[str, str],
    *,
    source_ids: Sequence[str],
    transform: str,
    factors: dict[str, str],
    duplicate_factor: int = 1,
) -> None:
    batch_dir = batch_root / batch_id
    _reset_flat_directory(batch_dir)
    artifacts = []
    for filename, text in sorted(documents.items()):
        payload = text.encode("utf-8")
        stem, suffix = Path(filename).stem, Path(filename).suffix
        for duplicate_index in range(duplicate_factor):
            output_name = filename if duplicate_index == 0 else f"{stem}__dup{duplicate_index + 1:02d}{suffix}"
            output_path = batch_dir / output_name
            output_path.write_bytes(payload)
            artifacts.append(
                {
                    "path": output_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "duplicate_of": filename if duplicate_index else None,
                }
            )
    _write_json(
        batch_dir / "_batch_manifest.json",
        {
            "schema_version": "cll-controlled-batch-v1",
            "batch_id": batch_id,
            "source_ids": list(source_ids),
            "transform": transform,
            "factors": factors,
            "duplicate_factor": duplicate_factor,
            "artifacts": artifacts,
        },
    )


def _simulate_parsing_noise(markdown: str) -> str:
    """Create a deterministic same-content parse-quality stress condition."""

    output = ["NCI CLL EXPORT | simulated parser output | page 1"]
    page = 1
    paragraph_count = 0
    for paragraph in re.split(r"\n\s*\n", markdown):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_count += 1
        if paragraph_count % 12 == 0:
            page += 1
            output.extend([f"NCI CLL EXPORT | simulated parser output | page {page}", str(page)])
        if paragraph.startswith("#"):
            output.append(paragraph.lstrip("# ").upper())
        elif paragraph.startswith("|"):
            output.append(re.sub(r"\s*\|\s*", "    ", paragraph).strip())
        else:
            flattened = re.sub(r"(?m)^-\s+", "", paragraph)
            output.extend(textwrap.wrap(flattened, width=72, break_long_words=False, break_on_hyphens=False))
    return "\n".join(output).rstrip() + "\n"


def _normalize_for_matching(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.replace("haemolytic", "hemolytic").replace("anaemia", "anemia").replace("leukaemia", "leukemia")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    if min(len(left_tokens), len(right_tokens)) < 4:
        return float(left == right)
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _classify_benchmark_item(question: str, answer: str, options: Sequence[str]) -> str:
    del options  # Distractor wording must not decide the capability label.
    text = " ".join([question, answer]).casefold()
    if re.search(r"\b(?:treat|therapy|agent|drug|inhibitor|chemotherap)", text):
        if re.search(r"\b(?:asymptomatic|watchful|withheld|initiat|indication|observation)", text):
            return "treatment_indication"
        return "treatment_selection"
    if re.search(r"\b(?:richter|transform|prognos|survival|life expectancy|17p|tp53|ighv|risk)", text):
        return "prognosis_and_transformation"
    if re.search(r"\b(?:infection|autoimmune|hemolytic|hypogammaglobulin|tumou?r lysis|complication)", text):
        return "complications"
    if re.search(r"\b(?:cd ?(?:5|19|20|23)|smudge|b[- ]?cell|lymphocyte|morpholog|immunophenotyp)", text):
        return "pathology_and_immunophenotype"
    return "diagnosis_and_differential"


def curate_medmcqa_cll_proxy(upstream_path: Path, output_path: Path) -> dict[str, Any]:
    """Create a CLL-focused, deterministic holdout from a pinned public split.

    MedMCQA's official test labels are hidden and its validation split contains
    only a few direct CLL questions. The proxy therefore draws from the pinned
    train split once, locks the resulting IDs, and never exposes these rows to
    source selection or SFT generation.
    """

    table = pq.read_table(upstream_path)
    option_fields = ("opa", "opb", "opc", "opd")
    candidates = []
    exclusion_counts: Counter[str] = Counter()
    for row in table.to_pylist():
        correct = row.get("cop")
        if not isinstance(correct, int) or not 0 <= correct < 4:
            exclusion_counts["invalid_answer_index"] += 1
            continue
        question = str(row.get("question") or "").strip()
        topic = str(row.get("topic_name") or "").strip()
        options = [str(row.get(field) or "").strip() for field in option_fields]
        answer = options[correct]
        exact_topic = _normalize_for_matching(topic) == "chronic lymphocytic leukemia"
        if not (CLL_PATTERN.search(question) or CLL_PATTERN.search(answer) or exact_topic):
            exclusion_counts["not_cll_targeted"] += 1
            continue
        combined = " ".join([question, *options])
        if "\ufffd" in combined:
            exclusion_counts["replacement_character"] += 1
            continue
        if len(question) < 12 or any(not option for option in options):
            exclusion_counts["missing_or_short_field"] += 1
            continue
        if len({_normalize_for_matching(option) for option in options}) < 4:
            exclusion_counts["duplicate_options"] += 1
            continue
        if re.fullmatch(r"(?i)[a-d]{1,4}", answer):
            exclusion_counts["malformed_combination_answer"] += 1
            continue
        quality_rank = (
            int(CLL_PATTERN.search(question) is not None),
            int(exact_topic),
            int(bool(row.get("exp"))),
            len(question),
        )
        candidates.append((quality_rank, row, options, answer))

    # Near-duplicate removal operates only inside benchmark curation. Downstream
    # data selection receives the blueprint file, never these stems or answers.
    retained: list[tuple[dict[str, Any], str]] = []
    for _, row, options, answer in sorted(candidates, key=lambda item: (tuple(-x for x in item[0]), item[1]["id"])):
        normalized_question = _normalize_for_matching(row["question"])
        if any(_jaccard(normalized_question, existing) >= 0.65 for _, existing in retained):
            exclusion_counts["near_duplicate"] += 1
            continue
        record = {
            "benchmark_item_id": f"medmcqa:{row['id']}",
            "source_dataset": "MedMCQA",
            "source_revision": MEDMCQA_COMMIT,
            "source_split": "train",
            "source_item_id": row["id"],
            "question": row["question"],
            "options": options,
            "answer_index": row["cop"],
            "answer_text": answer,
            "explanation": row.get("exp"),
            "subject_name": row.get("subject_name"),
            "topic_name": row.get("topic_name"),
            "choice_type": row.get("choice_type"),
            "capability_id": _classify_benchmark_item(row["question"], answer, options),
            "temporal_risk": bool(TEMPORAL_RISK_PATTERN.search(" ".join([row["question"], answer]))),
        }
        retained.append((record, normalized_question))

    records = sorted((record for record, _ in retained), key=lambda item: item["benchmark_item_id"])
    _write_jsonl(output_path, records)
    capability_counts = Counter(record["capability_id"] for record in records)
    return {
        "input_rows": table.num_rows,
        "candidate_rows_before_deduplication": len(candidates),
        "item_count": len(records),
        "capability_counts": dict(sorted(capability_counts.items())),
        "choice_type_counts": dict(sorted(Counter(record["choice_type"] for record in records).items())),
        "temporal_risk_item_count": sum(record["temporal_risk"] for record in records),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "payload_sha256": file_sha256(output_path),
    }


def _copy_source_config(source: SourceDefinition, raw_sha256: str) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "title": source.title,
        "uri": source.canonical_url,
        "version": source.version,
        "language": source.language,
        "license_status": "allowed",
        "allowed_uses": list(source.allowed_uses),
        "content_sha256": raw_sha256,
        "knowledge_cutoff": source.knowledge_cutoff,
    }


def _build_suite(
    config_dir: Path, batch_root: Path, source_records: dict[str, dict[str, Any]], blueprint_sha256: str
) -> Path:
    relative_batch_root = Path("..") / ".." / batch_root.relative_to(config_dir.parents[1])
    batch_specs = [
        (
            "B01",
            "factorial",
            "development",
            ["nci_cll_professional"],
            "controlled_pool",
            "broad",
            "high",
            "low",
            "clean",
        ),
        (
            "B02",
            "factorial",
            "development",
            ["nci_cll_professional"],
            "controlled_pool",
            "broad",
            "high",
            "high",
            "clean",
        ),
        (
            "B03",
            "factorial",
            "development",
            ["nci_aml_professional", "nci_cml_professional"],
            "controlled_pool",
            "broad",
            "low",
            "low",
            "clean",
        ),
        (
            "B04",
            "factorial",
            "development",
            ["nci_aml_professional", "nci_cml_professional"],
            "controlled_pool",
            "broad",
            "low",
            "high",
            "clean",
        ),
        (
            "B05",
            "factorial",
            "calibration",
            ["nci_cll_professional"],
            "controlled_pool",
            "narrow",
            "high",
            "low",
            "clean",
        ),
        (
            "B06",
            "factorial",
            "calibration",
            ["nci_cll_professional"],
            "controlled_pool",
            "narrow",
            "high",
            "high",
            "clean",
        ),
        (
            "B07",
            "factorial",
            "calibration",
            ["nci_aml_professional", "nci_cml_professional"],
            "controlled_pool",
            "narrow",
            "low",
            "low",
            "clean",
        ),
        (
            "B08",
            "factorial",
            "calibration",
            ["nci_aml_professional", "nci_cml_professional"],
            "controlled_pool",
            "narrow",
            "low",
            "high",
            "clean",
        ),
        (
            "B09",
            "natural_check",
            "calibration",
            ["nci_cll_professional"],
            "natural_pair_09_10",
            "natural",
            "unmeasured",
            "unmeasured",
            "clean",
        ),
        (
            "B10",
            "natural_check",
            "calibration",
            ["nci_cll_professional"],
            "natural_pair_09_10",
            "natural",
            "unmeasured",
            "unmeasured",
            "noisy",
        ),
        (
            "B11",
            "natural_check",
            "evaluation",
            ["pmc11171037"],
            "independent_pmc11171037",
            "natural",
            "unmeasured",
            "unmeasured",
            "natural",
        ),
        (
            "B12",
            "natural_check",
            "evaluation",
            ["nci_cll_patient", "pmc11505876"],
            "independent_mixed_12",
            "natural",
            "unmeasured",
            "unmeasured",
            "natural",
        ),
    ]
    batches = []
    for batch_id, kind, split, source_ids, lineage, coverage, relevance, redundancy, parsing in batch_specs:
        note = {
            "B01": "Six CLL capability units; no intentional duplication.",
            "B02": "The same six complete CLL units copied three times.",
            "B03": "Six size-matched AML/CML control units outside the CLL target.",
            "B04": "The same six AML/CML control units copied three times.",
            "B05": "Two complete CLL capability groups retained.",
            "B06": "The same two CLL capability groups copied three times.",
            "B07": "Two complete non-CLL leukemia control units retained.",
            "B08": "The same two non-CLL leukemia control units copied three times.",
            "B09": "Clean text-only extraction of the NCI professional CLL summary.",
            "B10": "B09 semantic content with deterministic line-wrap, header, and table-flattening noise.",
            "B11": "Independent CC BY review focused on CLL treatment sequencing.",
            "B12": "Natural mix of NCI patient material and an independent CC BY molecular review.",
        }[batch_id]
        entry = {
            "batch_id": batch_id,
            "kind": kind,
            "split": split,
            "input_uri": (relative_batch_root / batch_id).as_posix(),
            "source_ids": source_ids,
            "lineage_group": lineage,
            "factors": {
                "coverage_scope": coverage,
                "target_relevance": relevance,
                "redundancy": redundancy,
                "parsing_quality": parsing,
            },
            "construction_notes": [note],
        }
        if batch_id == "B09":
            entry["paired_with"] = ["B10"]
        elif batch_id == "B10":
            entry["paired_with"] = ["B09"]
        batches.append(entry)

    source_order = [source.source_id for source in PUBLIC_SOURCES]
    suite = {
        "schema_version": "batch-suite-v1",
        "suite_id": "cll-public-proxy-feasibility-12",
        "domain": "CLL/SLL foundational clinical knowledge; public feasibility proxy",
        "benchmark_id": "medmcqa-cll-proxy-v1",
        "intended_claim": "feasibility_and_method_screening",
        "statistical_unit": "batch",
        "capability_blueprint_uri": "capability_blueprint.yaml",
        "capability_blueprint_sha256": blueprint_sha256,
        "capability_blueprint": [capability["capability_id"] for capability in CAPABILITIES],
        "sources": [source_records[source_id]["suite_source"] for source_id in source_order],
        "batches": batches,
    }
    suite_path = config_dir / "batch_suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return suite_path


def _write_contract(config_dir: Path, example_dir: Path, blueprint_sha256: str, manifest_sha256: str) -> Path:
    contract = yaml.safe_load((example_dir / "production_contract.yaml").read_text(encoding="utf-8"))
    contract["contract_id"] = "cll-public-proxy-feasibility"
    contract["version"] = SNAPSHOT_DATE
    contract["benchmark"].update(
        {
            "benchmark_id": "medmcqa-cll-proxy-v1",
            "version": f"medmcqa-{MEDMCQA_COMMIT[:12]}-cll-proxy-v1",
            "capability_blueprint_sha256": blueprint_sha256,
            "test_manifest_sha256": manifest_sha256,
        }
    )
    path = config_dir / "production_contract.yaml"
    path.write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return path


def acquire_cll_feasibility_package(
    project_root: str | Path,
    *,
    refresh: bool = False,
    keep_benchmark_upstream: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    public_root = project_root / "data" / "acquired" / "valuation_cll"
    benchmark_root = project_root / "data" / "private" / "valuation_cll"
    config_dir = project_root / "configs" / "valuation_cll_public_feasibility"
    example_dir = project_root / "examples" / "valuation_cll"
    raw_dir, processed_dir, batch_root = public_root / "raw", public_root / "processed", public_root / "batches"
    license_dir = public_root / "license_evidence"
    for path in (raw_dir, processed_dir, batch_root, license_dir, benchmark_root, config_dir):
        path.mkdir(parents=True, exist_ok=True)

    license_evidence = []
    license_downloads = (
        (
            "nci_reuse_policy",
            ("https://www.cancer.gov/policies/copyright-reuse",),
            license_dir / "nci_reuse_policy.html",
        ),
        (
            "medmcqa_official_repository_license",
            ("https://raw.githubusercontent.com/medmcqa/medmcqa/" f"{MEDMCQA_GITHUB_COMMIT}/LICENSE.md",),
            license_dir / "medmcqa_LICENSE.md",
        ),
        (
            "medmcqa_distribution_card",
            (
                "https://hf-mirror.com/datasets/openlifescienceai/medmcqa/resolve/" f"{MEDMCQA_COMMIT}/README.md",
                "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/" f"{MEDMCQA_COMMIT}/README.md",
            ),
            license_dir / "medmcqa_huggingface_README.md",
        ),
    )
    for evidence_id, urls, path in license_downloads:
        fetched_from = _download(urls, path, refresh=refresh)
        license_evidence.append(
            {
                "evidence_id": evidence_id,
                "path": path.relative_to(project_root).as_posix(),
                "sha256": file_sha256(path),
                "fetched_from": fetched_from,
                "candidate_urls": list(urls),
            }
        )

    source_records: dict[str, dict[str, Any]] = {}
    extracted: dict[str, tuple[SourceDefinition, str, list[dict[str, str]], str]] = {}
    for source in PUBLIC_SOURCES:
        raw_path = raw_dir / source.raw_filename
        fetched_from = _download(source.download_urls, raw_path, refresh=refresh)
        payload = raw_path.read_bytes()
        if source.source_format == "nci_html":
            title, blocks = extract_nci_blocks(payload)
        else:
            title, blocks = extract_jats_blocks(payload)
        raw_hash = file_sha256(raw_path)
        markdown = _render_document(source, title, blocks, raw_hash)
        markdown_path = processed_dir / f"{source.source_id}.md"
        blocks_path = processed_dir / f"{source.source_id}.blocks.jsonl"
        markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
        _write_jsonl(blocks_path, blocks)
        extracted[source.source_id] = (source, markdown, blocks, raw_hash)
        source_records[source.source_id] = {
            **asdict(source),
            "download_urls": list(source.download_urls),
            "allowed_uses": list(source.allowed_uses),
            "fetched_from": fetched_from,
            "raw_path": raw_path.relative_to(project_root).as_posix(),
            "raw_sha256": raw_hash,
            "processed_markdown_path": markdown_path.relative_to(project_root).as_posix(),
            "processed_markdown_sha256": file_sha256(markdown_path),
            "blocks_path": blocks_path.relative_to(project_root).as_posix(),
            "block_count": len(blocks),
            "suite_source": _copy_source_config(source, raw_hash),
        }

    cll_source, _, cll_blocks, _ = extracted["nci_cll_professional"]
    relevant_units = _build_relevant_units(cll_source, cll_blocks)
    target_sizes = [len(relevant_units[capability["capability_id"]]) for capability in CAPABILITIES]
    irrelevant_units = _build_irrelevant_units(
        [
            (extracted["nci_aml_professional"][0], extracted["nci_aml_professional"][2]),
            (extracted["nci_cml_professional"][0], extracted["nci_cml_professional"][2]),
        ],
        target_sizes,
    )
    relevant_docs = {f"{index:02d}_{key}.md": value for index, (key, value) in enumerate(relevant_units.items(), 1)}
    irrelevant_docs = {f"{index:02d}_{key}.md": value for index, (key, value) in enumerate(irrelevant_units.items(), 1)}
    relevant_narrow = dict(list(relevant_docs.items())[:2])
    irrelevant_narrow = dict(list(irrelevant_docs.items())[:2])
    factors = {
        "B01": {"coverage_scope": "broad", "target_relevance": "high", "redundancy": "low"},
        "B02": {"coverage_scope": "broad", "target_relevance": "high", "redundancy": "high"},
        "B03": {"coverage_scope": "broad", "target_relevance": "low", "redundancy": "low"},
        "B04": {"coverage_scope": "broad", "target_relevance": "low", "redundancy": "high"},
        "B05": {"coverage_scope": "narrow", "target_relevance": "high", "redundancy": "low"},
        "B06": {"coverage_scope": "narrow", "target_relevance": "high", "redundancy": "high"},
        "B07": {"coverage_scope": "narrow", "target_relevance": "low", "redundancy": "low"},
        "B08": {"coverage_scope": "narrow", "target_relevance": "low", "redundancy": "high"},
    }
    _write_batch(
        batch_root,
        "B01",
        relevant_docs,
        source_ids=["nci_cll_professional"],
        transform="capability_units",
        factors=factors["B01"],
    )
    _write_batch(
        batch_root,
        "B02",
        relevant_docs,
        source_ids=["nci_cll_professional"],
        transform="capability_units_with_exact_copies",
        factors=factors["B02"],
        duplicate_factor=3,
    )
    _write_batch(
        batch_root,
        "B03",
        irrelevant_docs,
        source_ids=["nci_aml_professional", "nci_cml_professional"],
        transform="size_matched_non_cll_units",
        factors=factors["B03"],
    )
    _write_batch(
        batch_root,
        "B04",
        irrelevant_docs,
        source_ids=["nci_aml_professional", "nci_cml_professional"],
        transform="size_matched_non_cll_units_with_exact_copies",
        factors=factors["B04"],
        duplicate_factor=3,
    )
    _write_batch(
        batch_root,
        "B05",
        relevant_narrow,
        source_ids=["nci_cll_professional"],
        transform="two_capability_units",
        factors=factors["B05"],
    )
    _write_batch(
        batch_root,
        "B06",
        relevant_narrow,
        source_ids=["nci_cll_professional"],
        transform="two_capability_units_with_exact_copies",
        factors=factors["B06"],
        duplicate_factor=3,
    )
    _write_batch(
        batch_root,
        "B07",
        irrelevant_narrow,
        source_ids=["nci_aml_professional", "nci_cml_professional"],
        transform="two_non_cll_units",
        factors=factors["B07"],
    )
    _write_batch(
        batch_root,
        "B08",
        irrelevant_narrow,
        source_ids=["nci_aml_professional", "nci_cml_professional"],
        transform="two_non_cll_units_with_exact_copies",
        factors=factors["B08"],
        duplicate_factor=3,
    )
    _write_batch(
        batch_root,
        "B09",
        {"nci_cll_professional.md": extracted["nci_cll_professional"][1]},
        source_ids=["nci_cll_professional"],
        transform="clean_text_extraction",
        factors={"coverage_scope": "natural", "target_relevance": "unmeasured", "redundancy": "unmeasured"},
    )
    _write_batch(
        batch_root,
        "B10",
        {"nci_cll_professional_noisy.txt": _simulate_parsing_noise(extracted["nci_cll_professional"][1])},
        source_ids=["nci_cll_professional"],
        transform="deterministic_parser_noise",
        factors={"coverage_scope": "natural", "target_relevance": "unmeasured", "redundancy": "unmeasured"},
    )
    _write_batch(
        batch_root,
        "B11",
        {"pmc11171037.md": extracted["pmc11171037"][1]},
        source_ids=["pmc11171037"],
        transform="clean_text_extraction",
        factors={"coverage_scope": "natural", "target_relevance": "unmeasured", "redundancy": "unmeasured"},
    )
    _write_batch(
        batch_root,
        "B12",
        {
            "nci_cll_patient.md": extracted["nci_cll_patient"][1],
            "pmc11505876.md": extracted["pmc11505876"][1],
        },
        source_ids=["nci_cll_patient", "pmc11505876"],
        transform="natural_mixed_sources",
        factors={"coverage_scope": "natural", "target_relevance": "unmeasured", "redundancy": "unmeasured"},
    )

    upstream_path = benchmark_root / "upstream" / f"medmcqa-train-{MEDMCQA_COMMIT[:12]}.parquet"
    benchmark_canonical_url = (
        "https://huggingface.co/datasets/openlifescienceai/medmcqa/resolve/"
        f"{MEDMCQA_COMMIT}/data/train-00000-of-00001.parquet"
    )
    benchmark_fetch_url = benchmark_canonical_url.replace("https://huggingface.co", "https://hf-mirror.com")
    benchmark_fetched_from = _download(
        (benchmark_fetch_url, benchmark_canonical_url),
        upstream_path,
        refresh=refresh,
        expected_sha256=MEDMCQA_TRAIN_SHA256,
    )
    benchmark_payload = benchmark_root / "benchmark_locked" / "medmcqa_cll_proxy_v1.jsonl"
    benchmark_stats = curate_medmcqa_cll_proxy(upstream_path, benchmark_payload)

    blueprint = {
        "schema_version": "capability-blueprint-v1",
        "benchmark_id": "medmcqa-cll-proxy-v1",
        "test_items_included": False,
        "source": "High-level capability labels derived once during locked MedMCQA proxy curation.",
        "capabilities": [
            {key: capability[key] for key in ("capability_id", "description", "weight")} for capability in CAPABILITIES
        ],
    }
    blueprint_path = config_dir / "capability_blueprint.yaml"
    blueprint_path.write_text(yaml.safe_dump(blueprint, allow_unicode=True, sort_keys=False), encoding="utf-8")
    blueprint_sha256 = file_sha256(blueprint_path)

    benchmark_manifest = {
        "schema_version": "locked-benchmark-manifest-v1",
        "benchmark_id": "medmcqa-cll-proxy-v1",
        "version": f"medmcqa-{MEDMCQA_COMMIT[:12]}-cll-proxy-v1",
        "role": "test",
        "purpose": "Public feasibility proxy; replace with the user's benchmark for the valuation claim.",
        "item_payload_in_repository": False,
        "payload_uri": "../../data/private/valuation_cll/benchmark_locked/medmcqa_cll_proxy_v1.jsonl",
        "payload_sha256": benchmark_stats["payload_sha256"],
        "item_count": benchmark_stats["item_count"],
        "expected_item_id_policy": "unique-and-complete",
        "primary_metric": "accuracy",
        "upstream": {
            "dataset": "MedMCQA",
            "repository": "https://github.com/medmcqa/medmcqa",
            "distribution_repository": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
            "revision": MEDMCQA_COMMIT,
            "source_split": "train",
            "upstream_sha256": MEDMCQA_TRAIN_SHA256,
            "license_status": "audit_only",
            "license_note": (
                "The official GitHub repository labels its repository MIT while the distribution card labels the "
                "dataset Apache-2.0; underlying exam-item rights were not independently cleared. Use this locked "
                "payload for research evaluation only, not SFT generation."
            ),
            "license_evidence": [
                "data/acquired/valuation_cll/license_evidence/medmcqa_LICENSE.md",
                "data/acquired/valuation_cll/license_evidence/medmcqa_huggingface_README.md",
            ],
        },
        "curation": {
            "target_rule": "CLL/SLL appears in the stem, correct answer, or exact topic label.",
            "quality_rules": [
                "answer index is 0-3",
                "four non-empty distinct options",
                "no Unicode replacement character",
                "remove malformed combination-answer tokens",
                "greedy normalized-token Jaccard deduplication at 0.65",
            ],
            **benchmark_stats,
        },
        "limitations": [
            "This is a deterministic holdout from MedMCQA train because official test labels are hidden and validation has too few direct CLL items.",
            "Public pretraining contamination is possible.",
            "Automatic curation does not replace hematologist review; temporal-risk items are flagged in the payload.",
            "The payload is never an input to batch construction, mining, generation, or proxy selection.",
        ],
    }
    benchmark_manifest_path = config_dir / "benchmark_test_manifest.json"
    _write_json(benchmark_manifest_path, benchmark_manifest)

    suite_path = _build_suite(config_dir, batch_root, source_records, blueprint_sha256)
    frozen_suite_path = config_dir / "batch_suite.frozen.yaml"
    frozen_suite_path.unlink(missing_ok=True)
    freeze_manifest = freeze_batch_suite(suite_path, frozen_suite_path)
    contract_path = _write_contract(
        config_dir,
        example_dir,
        blueprint_sha256,
        file_sha256(benchmark_manifest_path),
    )

    portable_freeze_manifest = {
        **freeze_manifest,
        "frozen_file": frozen_suite_path.relative_to(project_root).as_posix(),
    }
    acquisition_manifest = {
        "schema_version": "cll-public-acquisition-v1",
        "snapshot_date": SNAPSHOT_DATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(source_records),
        "license_evidence": license_evidence,
        "sources": [
            {key: value for key, value in row.items() if key != "suite_source"} for row in source_records.values()
        ],
        "benchmark": {
            "canonical_url": benchmark_canonical_url,
            "fetched_from": benchmark_fetched_from,
            "upstream_path": upstream_path.relative_to(project_root).as_posix(),
            "upstream_sha256": file_sha256(upstream_path),
            "payload_path": benchmark_payload.relative_to(project_root).as_posix(),
            **benchmark_stats,
        },
        "batch_suite": {
            "source_path": suite_path.relative_to(project_root).as_posix(),
            "frozen_path": frozen_suite_path.relative_to(project_root).as_posix(),
            **portable_freeze_manifest,
        },
        "production_contract": {
            "path": contract_path.relative_to(project_root).as_posix(),
            "sha256": file_sha256(contract_path),
        },
    }
    acquisition_manifest_path = public_root / "acquisition_manifest.json"
    _write_json(acquisition_manifest_path, acquisition_manifest)
    # The reviewed lock is content-addressed and portable. Transport details
    # such as a mirror URL or cache hit remain in the runtime manifest only.
    acquisition_lock = json.loads(json.dumps(acquisition_manifest))
    acquisition_lock.pop("created_at_utc")
    acquisition_lock["benchmark"].pop("fetched_from", None)
    for row in acquisition_lock["sources"]:
        row.pop("fetched_from", None)
    for row in acquisition_lock["license_evidence"]:
        row.pop("fetched_from", None)
    _write_json(config_dir / "acquisition_lock.json", acquisition_lock)
    if not keep_benchmark_upstream:
        upstream_path.unlink(missing_ok=True)

    return {
        "public_data_root": str(public_root),
        "benchmark_payload": str(benchmark_payload),
        "benchmark_items": benchmark_stats["item_count"],
        "benchmark_payload_sha256": benchmark_stats["payload_sha256"],
        "batch_suite": str(suite_path),
        "frozen_batch_suite": str(frozen_suite_path),
        "suite_sha256": freeze_manifest["suite_sha256"],
        "production_contract": str(contract_path),
        "source_count": len(source_records),
    }
