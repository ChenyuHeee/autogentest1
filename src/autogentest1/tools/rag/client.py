"""RAG client backed by a persistent Chroma vector store."""

from __future__ import annotations

import math
import re
import uuid
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - dependency resolution happens at runtime
    import chromadb  # type: ignore
except Exception as _exc:  # pragma: no cover - exercised when chromadb missing
    chromadb = None  # type: ignore
    _CHROMADB_IMPORT_ERROR = _exc
else:  # pragma: no cover - keep reference for debugging
    _CHROMADB_IMPORT_ERROR = None


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenisation compatible with ascii-only corpora."""

    return _TOKEN_PATTERN.findall(text.lower())

class HashingEmbeddingFunction:
    """Deterministic hashing-based embedding function compatible with Chroma."""

    def __init__(self, *, dimensions: int = 1024) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in input:
            counts = [0.0] * self.dimensions
            for token in _tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % self.dimensions
                counts[bucket] += 1.0
            norm = math.sqrt(sum(value * value for value in counts))
            if norm > 0.0:
                counts = [value / norm for value in counts]
            vectors.append(counts)
        return vectors

    def name(self) -> str:
        """Return the identifier expected by newer Chroma releases."""

        return "autogentest1-hashing"

    def embed_text(self, text: str) -> List[float]:
        """Convenience wrapper that returns a single normalised vector."""

        return self([text])[0]

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:  # pragma: no cover - thin wrapper
        return self(input)

    def embed_query(self, input: Sequence[str]) -> List[List[float]]:  # pragma: no cover - thin wrapper
        return self(input)


@dataclass
class RagConfig:
    """RAG service configuration and file-system integration hints."""

    index_root: Path
    namespace: str = "default"
    embedding_dimensions: int = 1024
    chunk_size: int = 200
    overlap: int = 25
    distance_metric: str = "cosine"
    similarity_threshold: float = 0.1

    def ensure_directories(self) -> None:
        """Create index directories so ingestion scripts can write artifacts."""

        self.index_root.mkdir(parents=True, exist_ok=True)


@dataclass
class RagDocument:
    """Structured input payload for ingestion."""

    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _StoredChunk:
    """Internal representation used during ingestion."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    fingerprint: str


@dataclass
class RagQueryResult:
    """Structured payload returned from the RAG service."""

    question: str
    passages: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: List[dict] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)

    def top_passage(self) -> Optional[str]:
        """Convenience accessor for the highest-ranked passage."""

        return self.passages[0] if self.passages else None


class _VectorStore(ABC):
    """Minimal abstraction across vector store implementations."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[_StoredChunk]) -> int:
        """Persist new chunks and return the number actually stored."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""

    @abstractmethod
    def list_sources(self) -> List[str]:
        """Return sorted unique sources."""

    @abstractmethod
    def query(self, question: str, *, limit: int, threshold: float) -> RagQueryResult:
        """Execute a similarity search."""


def _compute_chunk_fingerprint(text: str, metadata: Dict[str, Any]) -> str:
    payload = {
        "text": text,
        "source": metadata.get("source"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_count": metadata.get("chunk_count"),
    }
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()


_CHUNK_UUID_NAMESPACE = uuid.UUID("7be535bf-9bb1-4a0f-9a95-5ad79cd5bd09")


def _chunk_id_from_fingerprint(fingerprint: str) -> str:
    return uuid.uuid5(_CHUNK_UUID_NAMESPACE, fingerprint).hex


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def _normalise_vector(vector: Sequence[float]) -> List[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


class _ChromadbStore(_VectorStore):
    """Wrapper around a persistent Chroma collection."""

    def __init__(self, config: RagConfig, embedding_function: HashingEmbeddingFunction) -> None:
        self._embedding_function = embedding_function
        self._client = chromadb.PersistentClient(path=str(config.index_root))  # type: ignore[attr-defined]
        self._collection = self._client.get_or_create_collection(
            name=config.namespace,
            metadata={"hnsw:space": config.distance_metric},
            embedding_function=self._embedding_function,
        )

    def add_chunks(self, chunks: Sequence[_StoredChunk]) -> int:
        if not chunks:
            return 0
        candidate_ids = [chunk.chunk_id for chunk in chunks]
        existing: set[str] = set()
        try:  # pragma: no cover - backend-specific behaviour
            lookup = self._collection.get(ids=candidate_ids, include=[])
        except Exception:
            lookup = {}
        raw_ids = lookup.get("ids") if isinstance(lookup, dict) else None
        if raw_ids:
            if raw_ids and isinstance(raw_ids[0], list):
                for item in raw_ids[0]:
                    if isinstance(item, str):
                        existing.add(item)
            else:
                for item in raw_ids:
                    if isinstance(item, str):
                        existing.add(item)

        filtered_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing]
        if not filtered_chunks:
            return 0

        try:
            self._collection.add(
                ids=[chunk.chunk_id for chunk in filtered_chunks],
                documents=[chunk.text for chunk in filtered_chunks],
                metadatas=[dict(chunk.metadata) for chunk in filtered_chunks],
            )
        except Exception:  # pragma: no cover - chroma raises on duplicates
            return 0
        return len(filtered_chunks)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover - backend-specific failure
            return 0

    def list_sources(self) -> List[str]:
        try:
            raw = self._collection.get(include=["metadatas"], limit=5000)
        except Exception:  # pragma: no cover - backend-specific failure
            return []
        entries = raw.get("metadatas") if isinstance(raw, dict) else None
        if entries and isinstance(entries, list) and entries and isinstance(entries[0], list):
            metadata_entries = entries[0]
        else:
            metadata_entries = entries or []
        sources = set()
        for entry in metadata_entries:
            if isinstance(entry, dict):
                value = entry.get("source")
                if value:
                    sources.add(str(value))
        return sorted(sources)

    def query(self, question: str, *, limit: int, threshold: float) -> RagQueryResult:
        try:
            result = self._collection.query(
                query_texts=[question],
                n_results=max(1, min(limit, 50)),
                include=["documents", "metadatas", "distances", "ids"],
            )
        except Exception:  # pragma: no cover - backend-specific failure
            return RagQueryResult(question=question)

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        passages: List[str] = []
        scores: List[float] = []
        metadata: List[Dict[str, Any]] = []
        chunk_ids: List[str] = []
        for index, doc in enumerate(documents):
            if not isinstance(doc, str):
                continue
            meta = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            chunk_id = ids[index] if index < len(ids) else None
            if isinstance(distance, (int, float)):
                similarity = max(0.0, 1.0 - float(distance))
            else:
                similarity = 0.0
            if similarity < threshold:
                continue
            passages.append(doc)
            scores.append(similarity)
            metadata.append(dict(meta) if isinstance(meta, dict) else {})
            chunk_ids.append(str(chunk_id) if chunk_id is not None else "")

        return RagQueryResult(
            question=question,
            passages=passages,
            scores=scores,
            metadata=metadata,
            ids=chunk_ids,
        )


@dataclass
class _JsonChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    fingerprint: str
    vector: List[float]


class _JsonVectorStore(_VectorStore):
    """Lightweight JSON-backed vector store used as a fallback."""

    def __init__(self, config: RagConfig, embedding_function: HashingEmbeddingFunction) -> None:
        self._config = config
        self._embedding_function = embedding_function
        self._path = config.index_root / f"{config.namespace}.json"
        self._chunks: List[_JsonChunk] = []
        self._fingerprints: Dict[str, int] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(chunks, list):
            return
        for entry in chunks:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            metadata = entry.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            fingerprint = entry.get("fingerprint")
            if not isinstance(fingerprint, str):
                fingerprint = _compute_chunk_fingerprint(text, metadata)
            vector = entry.get("vector")
            if not isinstance(vector, list) or not vector or not isinstance(vector[0], (int, float)):
                vector = self._embedding_function.embed_text(text)
            else:
                vector = _normalise_vector(vector)
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str):
                chunk_id = _chunk_id_from_fingerprint(fingerprint)
            metadata.setdefault("source", metadata.get("source", "unknown"))
            metadata.setdefault("fingerprint", fingerprint)
            json_chunk = _JsonChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                fingerprint=fingerprint,
                vector=vector,
            )
            self._fingerprints[fingerprint] = len(self._chunks)
            self._chunks.append(json_chunk)

    def _persist(self) -> None:
        payload = {
            "namespace": self._config.namespace,
            "embedding_model": self._embedding_function.name(),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "fingerprint": chunk.fingerprint,
                    "vector": chunk.vector,
                }
                for chunk in self._chunks
            ],
        }
        self._config.index_root.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def add_chunks(self, chunks: Sequence[_StoredChunk]) -> int:
        inserted = 0
        for chunk in chunks:
            if chunk.fingerprint in self._fingerprints:
                continue
            metadata = dict(chunk.metadata)
            metadata.setdefault("source", metadata.get("source", "unknown"))
            metadata.setdefault("fingerprint", chunk.fingerprint)
            vector = self._embedding_function.embed_text(chunk.text)
            json_chunk = _JsonChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=metadata,
                fingerprint=chunk.fingerprint,
                vector=vector,
            )
            self._fingerprints[chunk.fingerprint] = len(self._chunks)
            self._chunks.append(json_chunk)
            inserted += 1
        if inserted:
            self._persist()
        return inserted

    def count(self) -> int:
        return len(self._chunks)

    def list_sources(self) -> List[str]:
        sources = {chunk.metadata.get("source", "unknown") for chunk in self._chunks}
        return sorted(str(source) for source in sources if source)

    def query(self, question: str, *, limit: int, threshold: float) -> RagQueryResult:
        if not self._chunks:
            return RagQueryResult(question=question)
        question_vector = self._embedding_function.embed_text(question)
        scored: List[tuple[float, _JsonChunk]] = []
        for chunk in self._chunks:
            similarity = _cosine_similarity(question_vector, chunk.vector)
            if similarity < threshold:
                continue
            scored.append((similarity, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        limited = scored[: max(1, min(limit, 50))]
        return RagQueryResult(
            question=question,
            passages=[chunk.text for _, chunk in limited],
            scores=[score for score, _ in limited],
            metadata=[dict(chunk.metadata) for _, chunk in limited],
            ids=[chunk.chunk_id for _, chunk in limited],
        )


class RagService:
    """Interface to a persistent vector store with JSON fallback."""

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self._embedding_function = HashingEmbeddingFunction(dimensions=self.config.embedding_dimensions)
        self._store: _VectorStore
        if chromadb is not None:
            try:
                self._store = _ChromadbStore(config, self._embedding_function)
            except Exception:  # pragma: no cover - fallback when client setup fails
                self._store = _JsonVectorStore(config, self._embedding_function)
        else:
            self._store = _JsonVectorStore(config, self._embedding_function)

    # ------------------------------------------------------------------
    # Ingestion & querying
    # ------------------------------------------------------------------
    def ingest_documents(self, documents: Iterable[Any]) -> int:
        """Ingest documents into the vector store and return the chunk count."""

        prepared: List[_StoredChunk] = []
        seen_batch: set[str] = set()
        for raw in documents:
            document = self._coerce_document(raw)
            if document is None:
                continue
            chunks = self._chunk_text(document.body)
            if not chunks:
                continue
            total_chunks = len(chunks)
            for idx, text in enumerate(chunks):
                metadata = dict(document.metadata)
                metadata.setdefault("source", metadata.get("source", "unknown"))
                metadata.update({
                    "chunk_index": idx,
                    "chunk_count": total_chunks,
                })
                fingerprint = _compute_chunk_fingerprint(text, metadata)
                if fingerprint in seen_batch:
                    continue
                seen_batch.add(fingerprint)
                chunk_id = _chunk_id_from_fingerprint(fingerprint)
                metadata = dict(metadata)
                metadata["fingerprint"] = fingerprint
                prepared.append(
                    _StoredChunk(
                        chunk_id=chunk_id,
                        text=text,
                        metadata=metadata,
                        fingerprint=fingerprint,
                    )
                )

        if not prepared:
            return 0

        return self._store.add_chunks(prepared)

    def _coerce_document(self, raw: Any) -> Optional[RagDocument]:
        if raw is None:
            return None
        if isinstance(raw, RagDocument):
            return raw
        if isinstance(raw, Path):
            text = raw.read_text(encoding="utf-8").strip()
            if not text:
                return None
            return RagDocument(body=text, metadata={"source": str(raw)})
        if isinstance(raw, dict):
            body = str(raw.get("body", ""))
            if not body.strip():
                return None
            metadata = {key: value for key, value in raw.items() if key != "body"}
            return RagDocument(body=body, metadata=metadata)
        if isinstance(raw, str):
            cleaned = raw.strip()
            if not cleaned:
                return None
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return self._coerce_document(parsed)
            except json.JSONDecodeError:
                pass
            return RagDocument(body=cleaned, metadata={"source": "raw_text"})
        return None

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, min(self.config.overlap, chunk_size - 1))
        step = max(1, chunk_size - overlap)
        chunks: List[str] = []
        for start in range(0, len(words), step):
            slice_words = words[start : start + chunk_size]
            if not slice_words:
                continue
            chunks.append(" ".join(slice_words))
        return chunks or [" ".join(words)]

    def query(self, question: str, *, top_k: int = 5) -> RagQueryResult:
        """Return the closest matching passages for the supplied question."""

        if self.count() == 0:
            return RagQueryResult(question=question)

        limit = max(1, min(top_k, 50))
        threshold = max(0.0, float(self.config.similarity_threshold))
        return self._store.query(question, limit=limit, threshold=threshold)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def count(self) -> int:
        """Return the number of stored chunks (useful for diagnostics)."""

        return self._store.count()

    def list_sources(self) -> List[str]:
        """Return unique sources present in the current namespace."""

        return self._store.list_sources()
