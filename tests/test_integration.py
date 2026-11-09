from pathlib import Path

import numpy as np
import pytest

from ai_ready.config import EmbeddingConfig, ParsingConfig, load_config
from ai_ready.core.pipeline import L1Pipeline
from ai_ready.embedding.sentence_transformer import TextEmbedder
from ai_ready.format.documents import DocumentChunker, DocumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_bge_retrieval():
    embedder = TextEmbedder(EmbeddingConfig(device="cpu"))
    vectors = embedder.encode(["借款人提前还款需要提前30日提出申请。", "数据库每天做增量备份，每周做完整备份。"])
    query = embedder.encode(["办理提前还贷要提前几天申请？"], query=True)
    assert vectors.shape == (2, 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    scores = query @ vectors.T
    assert scores[0, 0] > scores[0, 1]


@pytest.mark.integration
def test_hybrid_chunker_fits_embedding_budget(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("贷款申请需要提供身份证明。" * 180, encoding="utf-8")
    parsing = ParsingConfig(max_tokens=96)
    embedding = EmbeddingConfig(device="cpu")
    doc, backend = DocumentParser(parsing).convert(path)
    _, units = DocumentChunker(parsing, embedding).chunk(doc, path, backend)
    assert len(units) > 1
    vectors = TextEmbedder(embedding).encode([unit.embedding_text for unit in units])
    assert len(vectors) == len(units)


@pytest.mark.integration
def test_complete_pipeline_and_reload_search(tmp_path):
    config = load_config(PROJECT_ROOT / "configs/l1.yaml")
    config.output_dir = tmp_path / "run"
    pipeline = L1Pipeline(config)
    manifest = pipeline.run()
    assert manifest["status"] == "completed"
    assert manifest["stages"]["parse"]["files_parsed"] == 2
    hits = pipeline.search("提前还贷需要提前多少天申请？", 2)
    assert "30日" in hits[0]["text"]
    assert hits[0]["sources"][0]["item_refs"]
    assert (config.output_dir / "clusters/assignments.jsonl").is_file()
    assert (config.output_dir / "documents.parquet").is_file()
    assert (config.output_dir / "document_units.parquet").is_file()
    assert (config.output_dir / "topics.parquet").is_file()
    assert (config.output_dir / "demand_cells.parquet").is_file()
    assert (config.output_dir / "selected_anchors.parquet").is_file()
    assert (config.output_dir / "knowledge_edges.parquet").is_file()
    assert (config.output_dir / "catalog.duckdb").is_file()
    assert (config.output_dir / "snapshot.json").is_file()
    assert manifest["mlflow_run_id"]


@pytest.mark.integration
@pytest.mark.parametrize("file_format", ["pdf", "png"])
def test_real_docling_scanned_pdf_ocr(tmp_path, file_format):
    import pymupdf

    path = tmp_path / f"scanned_policy.{file_format}"
    with pymupdf.open() as source:
        page = source.new_page()
        page.insert_text((72, 120), "LOAN POLICY", fontsize=24)
        page.insert_text((72, 180), "Submit the application 30 days in advance.", fontsize=16)
        page.insert_text((72, 230), "Minimum repayment amount is 10000 yuan.", fontsize=16)
        page.insert_text((72, 280), "提前还款需要提前30日申请。", fontsize=16, fontname="china-s")
        bitmap = page.get_pixmap(dpi=180).tobytes("png")
    if file_format == "pdf":
        with pymupdf.open() as scanned:
            page = scanned.new_page()
            page.insert_image(page.rect, stream=bitmap)
            scanned.save(path)
    else:
        path.write_bytes(bitmap)
    parsing = ParsingConfig(backend="auto", force_full_page_ocr=True, table_structure=False)
    document, backend = DocumentParser(parsing).convert(path)
    assert backend == "docling"
    text = document.export_to_markdown()
    assert "30" in text and "10000" in text
    assert "申请" in text
    _, units = DocumentChunker(ParsingConfig(chunker="hierarchical"), EmbeddingConfig()).chunk(document, path, backend)
    assert any(source.locations for unit in units for source in unit.sources)


@pytest.mark.integration
def test_real_marker_digital_pdf_adapter(tmp_path):
    import pymupdf

    path = tmp_path / "complex_policy.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Loan Policy", fontsize=20)
        page.insert_text((72, 120), "Article 12: Apply 30 days in advance.", fontsize=12)
        page.insert_text((320, 120), "Article 15: Minimum 10000 yuan.", fontsize=12)
        pdf.save(path)
    document, backend = DocumentParser(
        ParsingConfig(backend="marker", marker_mode="fast", table_structure=True)
    ).convert(path)
    text = document.export_to_markdown()
    assert backend == "marker-fast/docling"
    assert "30" in text and "10000" in text
