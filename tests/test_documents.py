import pytest

from ai_ready.config import EmbeddingConfig, ParsingConfig
from ai_ready.format.documents import DocumentChunker, DocumentParser


def test_docx_retains_order_headings_and_table(tmp_path):
    from docx import Document

    path = tmp_path / "条款.docx"
    source = Document()
    source.add_heading("贷款管理", level=1)
    source.add_paragraph("客户必须提前30日提交申请。")
    table = source.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "贷款类型"
    table.cell(0, 1).text = "最低金额"
    table.cell(1, 0).text = "商业贷款"
    table.cell(1, 1).text = "10000元"
    source.add_paragraph("公积金贷款另行规定。")
    source.save(path)
    doc, parser_name = DocumentParser(ParsingConfig()).convert(path)
    markdown = doc.export_to_markdown()
    assert markdown.index("30日") < markdown.index("10000元") < markdown.index("另行规定")
    assert len(doc.tables) == 1
    chunker = DocumentChunker(ParsingConfig(chunker="hierarchical"), EmbeddingConfig())
    _, units = chunker.chunk(doc, path, parser_name)
    assert all(unit.sources[0].item_refs for unit in units)
    assert any("贷款管理" in unit.headings for unit in units)
    assert all(not source.locations for unit in units for source in unit.sources)


def test_pdf_retains_actual_page_and_bbox(tmp_path):
    import pymupdf

    path = tmp_path / "原文.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Rule A: apply 30 days in advance.")
        pdf.new_page().insert_text((72, 72), "Rule B: repay at least 10000 yuan.")
        pdf.save(path)
    document, backend = DocumentParser(ParsingConfig(backend="native", table_structure=False)).convert(path)
    chunker = DocumentChunker(ParsingConfig(chunker="hierarchical"), EmbeddingConfig())
    document_id, units = chunker.chunk(document, path, backend)
    assert {loc["page_no"] for unit in units for loc in unit.sources[0].locations} == {1, 2}
    assert all(loc["bbox"]["coord_origin"] == "TOPLEFT" for unit in units for loc in unit.sources[0].locations)
    assert chunker.chunk(document, path, backend)[0] == document_id


def test_txt_ids_change_with_content_and_empty_fails(tmp_path):
    path = tmp_path / "source.txt"
    parser = DocumentParser(ParsingConfig())
    chunker = DocumentChunker(ParsingConfig(chunker="hierarchical"), EmbeddingConfig())
    path.write_text("允许提前还款。", encoding="utf-8")
    doc, backend = parser.convert(path)
    first_id, first_units = chunker.chunk(doc, path, backend)
    path.write_text("不允许提前还款。", encoding="utf-8")
    doc, backend = parser.convert(path)
    second_id, second_units = chunker.chunk(doc, path, backend)
    assert first_id == second_id  # logical document identity is stable across versions
    assert first_units[0].document_version_id != second_units[0].document_version_id
    path.write_text("", encoding="utf-8")
    doc, backend = parser.convert(path)
    with pytest.raises(ValueError, match="No usable text"):
        chunker.chunk(doc, path, backend)


def test_scanned_pdf_without_ocr_fails(tmp_path):
    import pymupdf

    path = tmp_path / "scan.pdf"
    with pymupdf.open() as original:
        page = original.new_page()
        page.insert_text((72, 72), "Scanned source document")
        image = page.get_pixmap(dpi=150).tobytes("png")
    with pymupdf.open() as scanned:
        page = scanned.new_page()
        page.insert_image(page.rect, stream=image)
        scanned.save(path)
    with pytest.raises(ValueError, match="OCR is disabled"):
        DocumentParser(ParsingConfig(ocr=False)).convert(path)


def test_docling_markdown_adapter(tmp_path):
    path = tmp_path / "规则.md"
    path.write_text("# 贷款\n\n## 条件\n\n必须提前30日申请。\n", encoding="utf-8")
    doc, backend = DocumentParser(ParsingConfig(backend="docling")).convert(path)
    assert backend == "docling"
    assert "30日" in doc.export_to_markdown()


def test_html_and_window_copies_keep_original_gold_units(tmp_path):
    path = tmp_path / "规则.html"
    path.write_text(
        "<html><body><h1>贷款</h1><p>第一条：提前申请。</p><p>第二条：提交材料。</p></body></html>",
        encoding="utf-8",
    )
    parsing = ParsingConfig(chunker="hierarchical", sliding_window_units=2)
    doc, backend = DocumentParser(parsing).convert(path)
    _, units = DocumentChunker(parsing, EmbeddingConfig()).chunk(doc, path, backend)
    originals = [unit for unit in units if not unit.is_window_copy]
    windows = [unit for unit in units if unit.is_window_copy]
    assert "提前申请" in doc.export_to_markdown()
    assert windows
    assert windows[0].gold_source_unit_ids == [unit.id for unit in originals[:2]]
    assert all(unit.gold_source_unit_ids == [unit.id] for unit in originals)
