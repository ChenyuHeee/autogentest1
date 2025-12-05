"""Tests for the lightweight RAG client and persistence layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from autogentest1.config.settings import Settings
from autogentest1.tools import rag_tools
from autogentest1.tools.rag import RagConfig, RagDocument, RagService
from autogentest1.tools.rag import client as rag_client

@pytest.fixture(autouse=True)
def _force_json_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_client, "chromadb", None)


def test_rag_ingest_and_query(tmp_path: Path) -> None:
    config = RagConfig(index_root=tmp_path, namespace="macro")
    service = RagService(config)

    documents = [
        RagDocument(
            body="Paul Volcker engineered a rapid tightening in 1979 that initially pushed gold to new highs before real rates recovered.",
            metadata={"event": "Volcker tightening", "year": 1979, "category": "policy"},
        ),
        RagDocument(
            body="The 2013 taper tantrum combined rising real yields with a stronger dollar, suppressing gold prices despite growth worries.",
            metadata={"event": "Taper tantrum", "year": 2013, "category": "liquidity"},
        ),
    ]

    chunk_count = service.ingest_documents(documents)
    assert chunk_count > 0
    assert service.count() == chunk_count

    result = service.query("What happened to gold when Volcker tightened policy in 1979?", top_k=2)
    assert result.passages
    assert any("volcker" in passage.lower() for passage in result.passages)
    assert result.metadata and result.metadata[0]["event"] == "Volcker tightening"
    assert result.scores and result.scores[0] > service.config.similarity_threshold
    assert result.ids

    # Re-open the service to ensure persistence works.
    service_reloaded = RagService(config)
    assert service_reloaded.count() == service.count()
    assert service_reloaded.list_sources()  # should surface stored metadata


def test_rag_handles_unknown_query(tmp_path: Path) -> None:
    config = RagConfig(index_root=tmp_path)
    service = RagService(config)

    service.ingest_documents([
        RagDocument(body="Gold rallied in 2020 when real yields collapsed under pandemic policies."),
    ])

    result = service.query("Explain Martian weather patterns", top_k=3)
    assert result.question.startswith("Explain Martian")
    assert result.passages == []
    assert result.metadata == []


def test_rag_corpus_ingestion(tmp_path: Path) -> None:
    corpus_root = Path(__file__).resolve().parents[1] / "data/rag"
    assert corpus_root.exists()

    documents = []
    source_paths = sorted(p for p in corpus_root.glob("**/*") if p.suffix.lower() in {".json", ".md", ".txt"})
    for path in source_paths:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            body = payload.pop("body", "")
            if not body:
                continue
            metadata = {"source": str(path)}
            metadata.update(payload)
            documents.append(RagDocument(body=body, metadata=metadata))
        else:
            body = path.read_text(encoding="utf-8").strip()
            if not body:
                continue
            documents.append(RagDocument(body=body, metadata={"source": str(path)}))

    config = RagConfig(index_root=tmp_path)
    service = RagService(config)
    count = service.ingest_documents(documents)
    assert count == service.count()

    result = service.query("How did gold behave during the taper tantrum?", top_k=3)
    assert result.passages
    assert any("taper" in passage.lower() for passage in result.passages)

    playbook_hit = service.query("Describe a mean reversion playbook for gold", top_k=3)
    assert playbook_hit.passages
    assert any("mean reversion" in passage.lower() for passage in playbook_hit.passages)


def test_rag_tools_query_auto_ingest(tmp_path: Path, monkeypatch) -> None:
    rag_tools.reset_rag_cache()

    class DummySettings:
        rag_index_root = str(tmp_path / "index")
        rag_namespace = "unit-test"
        rag_chunk_size = 128
        rag_chunk_overlap = 24
        rag_similarity_threshold = 0.1
        rag_auto_ingest = True
        rag_corpus_paths = [str(Path(__file__).resolve().parents[2] / "data" / "rag")]

    settings_obj = cast(Settings, DummySettings())

    monkeypatch.setattr(rag_tools, "get_settings", lambda: settings_obj)

    info = rag_tools.ensure_default_corpus_loaded(settings=settings_obj, force=True)
    assert info["documents"] > 0
    result = rag_tools.query_playbook("mean reversion in gold", top_k=2, settings=settings_obj)
    assert result["passages"]
    assert len(result["metadata"]) == len(result["passages"])


def test_rag_tools_auto_ingest_fallback(tmp_path: Path, monkeypatch) -> None:
    rag_tools.reset_rag_cache()

    class DummySettings:
        rag_index_root = str(tmp_path / "index")
        rag_namespace = "fallback"
        rag_chunk_size = 128
        rag_chunk_overlap = 24
        rag_similarity_threshold = 0.1
        rag_auto_ingest = True
        rag_corpus_paths = ["/path/that/does/not/exist"]

    settings_obj = cast(Settings, DummySettings())
    monkeypatch.setattr(rag_tools, "get_settings", lambda: settings_obj)

    info = rag_tools.ensure_default_corpus_loaded(settings=settings_obj, force=True)
    assert info["documents"] > 0


def test_rag_json_fallback_snapshot(tmp_path: Path) -> None:
    snapshot = Path(__file__).resolve().parents[1] / "data" / "rag-index" / "default.json"
    index_root = tmp_path / "index"
    index_root.mkdir(parents=True)
    target = index_root / "default.json"
    target.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")

    service = RagService(RagConfig(index_root=index_root, namespace="default"))
    assert service.count() > 0

    result = service.query("How did gold behave during the taper tantrum?", top_k=3)
    assert result.passages
    assert any("taper" in passage.lower() for passage in result.passages)
    assert result.ids


def test_rag_json_fallback_idempotent_ingest(tmp_path: Path) -> None:
    service = RagService(RagConfig(index_root=tmp_path, namespace="unit"))

    document = RagDocument(
        body="Gold rallied sharply when real yields collapsed during the pandemic.",
        metadata={"source": "unit-test", "year": 2020},
    )

    first = service.ingest_documents([document, document])
    assert first == 1
    baseline = service.count()

    second = service.ingest_documents([document])
    assert second == 0
    assert service.count() == baseline
