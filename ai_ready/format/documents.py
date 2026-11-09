"""Parse local documents with Docling or lightweight native format adapters."""

import hashlib
import os
import re
from pathlib import Path

from ai_ready.config import EmbeddingConfig, ParsingConfig
from ai_ready.core.schema import DocumentRecord, EvidenceUnit, Provenance, stable_id

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
}
REFERENCE_PATTERN = re.compile(
    r"第[一二三四五六七八九十百千万零〇\d]+条|(?:参见|见|依据|根据|按照)[^，。；\n]{1,40}|"
    r"(?:article|section|appendix)\s+[A-Za-z0-9._-]+",
    re.IGNORECASE,
)


class DocumentParser:
    def __init__(self, config: ParsingConfig):
        self.config = config
        self._converter = None
        self.last_attachment = None

    def convert(self, path: Path):
        from docling_core.types.doc import DocItemLabel, DoclingDocument

        suffix = path.suffix.lower()
        self.last_attachment = None
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {suffix}")
        if suffix == ".txt":
            doc = DoclingDocument(name=path.stem)
            for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig")):
                if paragraph.strip():
                    doc.add_text(label=DocItemLabel.TEXT, text=paragraph)
            return doc, "text/docling-core"
        if self.config.backend == "marker":
            if suffix != ".pdf":
                raise ValueError("Marker backend is restricted to PDF; use Docling for other formats")
            return self._marker(path)
        if self.config.backend == "docling" or suffix in {
            ".html",
            ".htm",
            ".md",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
        }:
            return self._docling(path)
        if suffix == ".docx":
            return self._docx(path), "python-docx/docling-core"
        if self._needs_ocr(path):
            if not self.config.ocr:
                raise ValueError(f"Scanned content detected but OCR is disabled: {path.name}")
            return self._docling(path)
        if self.config.backend == "auto" and self._is_complex_pdf(path):
            if self.config.complex_pdf_backend == "marker":
                return self._marker(path)
            return self._docling(path)
        return self._pdf(path), "pymupdf/docling-core"

    def _is_complex_pdf(self, path: Path) -> bool:
        """Conservative signal: tables or multiple separated text columns."""
        import pymupdf

        with pymupdf.open(path) as pdf:
            for page in pdf:
                if self.config.table_structure and page.find_tables().tables:
                    return True
                blocks = [block for block in page.get_text("blocks") if block[6] == 0 and block[4].strip()]
                left_edges = sorted({round(block[0] / 72) for block in blocks})
                if len(blocks) >= 6 and len(left_edges) >= 3:
                    return True
        return False

    def _needs_ocr(self, path: Path) -> bool:
        import pymupdf

        if self.config.force_full_page_ocr:
            return True
        with pymupdf.open(path) as pdf:
            for page in pdf:
                if page.get_images() and len(page.get_text().strip()) < 80:
                    return True
        return False

    def _docling(self, path: Path):
        from docling.datamodel.base_models import ConversionStatus, InputFormat
        from docling.datamodel.pipeline_options import (
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        if self._converter is None:
            options = PdfPipelineOptions()
            options.do_ocr = self.config.ocr
            options.do_table_structure = self.config.table_structure
            ocr_options = RapidOcrOptions()
            if cache_dir := os.environ.get("AI_READY_CACHE_DIR"):
                ocr_options.rapidocr_params = {"Global.model_root_dir": str(Path(cache_dir) / "rapidocr")}
            if self.config.force_full_page_ocr:
                ocr_options.mode = OcrMode.FULL_PAGE
            options.ocr_options = ocr_options
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
                }
            )
        result = self._converter.convert(path)
        if result.status != ConversionStatus.SUCCESS:
            raise ValueError(f"Docling conversion incomplete ({result.status}): {path.name}; {result.errors}")
        return result.document, "docling"

    def _marker(self, path: Path):
        """Use Marker for complex digital PDFs, then normalize its Markdown through Docling."""
        import tempfile

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        parser = ConfigParser(
            {
                "output_format": "markdown",
                "mode": self.config.marker_mode,
                "disable_ocr": True,
                "disable_image_extraction": True,
            }
        )
        converter = PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=parser.get_processors(),
            renderer=parser.get_renderer(),
            llm_service=parser.get_llm_service(),
        )
        rendered = converter(str(path))
        markdown, metadata, _ = text_from_rendered(rendered)
        self.last_attachment = {
            "backend": "marker",
            "source": path.name,
            "metadata": metadata,
            "markdown": markdown,
        }
        cache_dir = Path(os.environ.get("AI_READY_CACHE_DIR", tempfile.gettempdir()))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".md", dir=cache_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(markdown)
        try:
            document, _ = self._docling(temporary)
            document.name = path.stem
        finally:
            temporary.unlink(missing_ok=True)
        return document, f"marker-{self.config.marker_mode}/docling"

    @staticmethod
    def _docx(path: Path):
        from docling_core.types.doc import (
            DocItemLabel,
            DoclingDocument,
            TableCell,
            TableData,
        )
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = DoclingDocument(name=path.stem)
        heading_stack = []
        for block in Document(path).iter_inner_content():
            parent = heading_stack[-1][1] if heading_stack else None
            if isinstance(block, Paragraph):
                if not block.text.strip():
                    continue
                style = block.style.name if block.style else ""
                match = re.match(r"Heading (\d+)", style)
                if match or style == "Title":
                    level = int(match.group(1)) if match else 1
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    parent = heading_stack[-1][1] if heading_stack else None
                    heading = doc.add_heading(text=block.text, level=level, parent=parent)
                    heading_stack.append((level, heading))
                else:
                    doc.add_text(label=DocItemLabel.TEXT, text=block.text, parent=parent)
            elif isinstance(block, Table):
                rows = [[cell.text for cell in row.cells] for row in block.rows]
                cells = [
                    TableCell(
                        text=text,
                        start_row_offset_idx=r,
                        end_row_offset_idx=r + 1,
                        start_col_offset_idx=c,
                        end_col_offset_idx=c + 1,
                        column_header=r == 0,
                    )
                    for r, row in enumerate(rows)
                    for c, text in enumerate(row)
                ]
                doc.add_table(
                    data=TableData(num_rows=len(rows), num_cols=max(map(len, rows), default=0), table_cells=cells),
                    parent=parent,
                )
        return doc

    def _pdf(self, path: Path):
        import pymupdf
        from docling_core.types.doc import (
            BoundingBox,
            CoordOrigin,
            DocItemLabel,
            DoclingDocument,
            ProvenanceItem,
            Size,
            TableCell,
            TableData,
        )

        doc = DoclingDocument(name=path.stem)
        with pymupdf.open(path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                doc.add_page(page_no=page_no, size=Size(width=page.rect.width, height=page.rect.height))
                tables = page.find_tables().tables if self.config.table_structure else []
                items = [(table.bbox[1], "table", table) for table in tables]
                for block in page.get_text("blocks", sort=True):
                    if block[6] != 0 or not block[4].strip():
                        continue
                    rect = pymupdf.Rect(block[:4])
                    if any(pymupdf.Rect(table.bbox).contains(rect) for table in tables):
                        continue
                    items.append((block[1], "text", block))
                for _, kind, item in sorted(items, key=lambda entry: entry[0]):
                    box = item.bbox if kind == "table" else item[:4]
                    text = "" if kind == "table" else item[4]
                    prov = ProvenanceItem(
                        page_no=page_no,
                        bbox=BoundingBox(l=box[0], t=box[1], r=box[2], b=box[3], coord_origin=CoordOrigin.TOPLEFT),
                        charspan=(0, len(text)),
                    )
                    if kind == "text":
                        doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
                    else:
                        rows = item.extract()
                        cells = [
                            TableCell(
                                text=value or "",
                                start_row_offset_idx=r,
                                end_row_offset_idx=r + 1,
                                start_col_offset_idx=c,
                                end_col_offset_idx=c + 1,
                                column_header=r == 0,
                            )
                            for r, row in enumerate(rows)
                            for c, value in enumerate(row)
                        ]
                        doc.add_table(
                            data=TableData(num_rows=item.row_count, num_cols=item.col_count, table_cells=cells),
                            prov=prov,
                        )
        return doc


class DocumentChunker:
    def __init__(self, parsing: ParsingConfig, embedding: EmbeddingConfig):
        from docling_core.transforms.chunker.hierarchical_chunker import (
            HierarchicalChunker,
        )

        self.parsing = parsing
        if parsing.chunker == "hybrid":
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                embedding.model_name, revision=embedding.revision, local_files_only=embedding.local_files_only
            )
            self.chunker = HybridChunker(
                tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=parsing.max_tokens), merge_peers=True
            )
        else:
            self.chunker = HierarchicalChunker()

    def chunk(
        self, document, path: Path, backend: str, registration: DocumentRecord | None = None
    ) -> tuple[str, list[EvidenceUnit]]:
        with path.open("rb") as handle:
            source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        source_uri = path.resolve().as_uri()
        document_id = registration.document_id if registration else stable_id("document", source_uri)
        document_version_id = (
            registration.document_version_id
            if registration
            else stable_id("document-version", document_id, source_hash)
        )
        units = []
        for ordinal, chunk in enumerate(self.chunker.chunk(dl_doc=document)):
            if not chunk.text.strip():
                continue
            headings = list(chunk.meta.headings or [])
            item_refs = [item.self_ref for item in chunk.meta.doc_items]
            labels = [item.label.value for item in chunk.meta.doc_items]
            if labels and all(label in {"page_header", "page_footer"} for label in labels):
                continue
            locations = [
                {"item_ref": item.self_ref, **prov.model_dump(mode="json")}
                for item in chunk.meta.doc_items
                for prov in item.prov
            ]
            provenance = Provenance(
                document_id=document_id,
                document_version_id=document_version_id,
                source_uri=source_uri,
                source_sha256=source_hash,
                parser=backend,
                headings=headings,
                item_refs=item_refs,
                locations=locations,
                raw_text=chunk.text,
            )
            units.append(
                EvidenceUnit(
                    id=stable_id(document_version_id, str(ordinal), "|".join(item_refs), chunk.text),
                    document_id=document_id,
                    document_version_id=document_version_id,
                    ordinal=ordinal,
                    unit_type=self._unit_type(labels),
                    text=chunk.text,
                    raw_text=chunk.text,
                    headings=headings,
                    sources=[provenance],
                    cross_references=sorted(set(REFERENCE_PATTERN.findall(chunk.text))),
                    structure_labels=sorted(set(labels)),
                )
            )
        if not units:
            raise ValueError(f"No usable text extracted from {path.name}; check OCR or document format")
        for unit in units:
            unit.gold_source_unit_ids = [unit.id]
        units.extend(self._window_copies(units, document_id, document_version_id))
        return document_id, units

    @staticmethod
    def _unit_type(labels: list[str]) -> str:
        priority = ("table", "list_item", "footnote", "formula", "code", "text")
        return next((label for label in priority if label in labels), labels[0] if labels else "text")

    def _window_copies(
        self, units: list[EvidenceUnit], document_id: str, document_version_id: str
    ) -> list[EvidenceUnit]:
        size = self.parsing.sliding_window_units
        if size < 2 or len(units) < size:
            return []
        windows = []
        for start in range(0, len(units) - size + 1, self.parsing.sliding_window_stride):
            members = units[start : start + size]
            source_ids = [item.id for item in members]
            text = "\n".join(item.text for item in members)
            raw_text = "\n".join(item.raw_text for item in members)
            sources = [source.model_copy(deep=True) for item in members for source in item.sources]
            windows.append(
                EvidenceUnit(
                    id=stable_id(document_version_id, "window", *source_ids),
                    document_id=document_id,
                    document_version_id=document_version_id,
                    ordinal=len(units) + start,
                    unit_type="window",
                    text=text,
                    raw_text=raw_text,
                    headings=members[0].headings,
                    sources=sources,
                    is_window_copy=True,
                    gold_source_unit_ids=source_ids,
                    cross_references=sorted({ref for item in members for ref in item.cross_references}),
                    structure_labels=sorted({label for item in members for label in item.structure_labels}),
                )
            )
        return windows
