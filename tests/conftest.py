from pathlib import Path

import pytest

from ai_ready.core.schema import EvidenceUnit, Provenance
from ai_ready.utils.runtime import configure_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
configure_runtime(PROJECT_ROOT / ".cache")


@pytest.fixture
def unit_factory():
    def make(identifier, text, headings=None):
        return EvidenceUnit(
            id=identifier,
            text=text,
            raw_text=text,
            headings=headings or [],
            sources=[
                Provenance(
                    document_id=f"doc-{identifier}",
                    source_uri=f"file:///sample-{identifier}.txt",
                    source_sha256=identifier,
                    parser="test",
                    raw_text=text,
                )
            ],
        )

    return make


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Run --run-integration to use real embedding/OCR models"))
