"""
================================================================================
agents/knowledge_base_agent.py  —  HR Knowledge Base Agent (RAG)
================================================================================

Purpose:
    Retrieval-Augmented Generation (RAG) engine over the company's HR document
    corpus.  Ingests HR policy documents (PDF, DOCX, TXT, Markdown) into a
    vector store, then retrieves the most relevant chunks for any employee
    question before generating a grounded answer.

    This is the pattern that replaces "search the SharePoint for the policy" —
    an employee asks a natural-language question and gets a precise, cited
    answer with the exact document and section that backs it up.

RAG pipeline:
    Ingestion:
        Document loader → text chunker → embedding model → vector store

    Retrieval:
        Query → embed → cosine similarity search → top-K chunks → rerank

    Generation:
        Chunks + query → LLM prompt → grounded answer + citations

Embedding strategy:
    Uses sentence-transformers (all-MiniLM-L6-v2) for local embeddings — no
    external API call required for the embedding step.  This is important for
    HR data which may contain sensitive policy details.

    Vector store: ChromaDB (SQLite-backed, runs in-process).
    In production: swap for Pinecone, Weaviate, or Azure AI Search.

Grounding guarantee:
    Every answer includes source_citations — the exact document name, page
    number (where available), and verbatim chunk excerpt.  If no relevant
    chunks are found the agent says so rather than hallucinating.

Usage:
    kb = KnowledgeBaseAgent()
    kb.ingest_document("data/hr_handbook.pdf", doc_type="policy")
    result = kb.query("How many sick days am I entitled to?")
    print(result.answer)
    print(result.citations)
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("platform.knowledge_base")

# ── Default document corpus path ─────────────────────────────────────────────
DEFAULT_CORPUS_PATH = "knowledge_base/documents"
DEFAULT_INDEX_PATH  = "knowledge_base/vector_index"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Output of one KnowledgeBaseAgent.query() call.

    Attributes:
        query:          The original employee question.
        answer:         Grounded answer generated from retrieved chunks.
        citations:      List of source references backing the answer.
        chunks_used:    The raw text chunks that were retrieved.
        confidence:     Retrieval confidence (0-1, based on similarity scores).
        found_relevant: True if at least one relevant chunk was retrieved.
        latency_ms:     End-to-end query latency.
    """
    query:          str
    answer:         str
    citations:      List[Dict[str, Any]]
    chunks_used:    List[Dict[str, Any]]
    confidence:     float
    found_relevant: bool
    latency_ms:     float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":          self.query,
            "answer":         self.answer,
            "citations":      self.citations,
            "confidence":     round(self.confidence, 3),
            "found_relevant": self.found_relevant,
            "latency_ms":     round(self.latency_ms, 2),
        }


@dataclass
class IngestResult:
    """Result of one document ingestion operation."""
    document_id:  str
    filename:     str
    chunks_added: int
    doc_type:     str
    success:      bool
    message:      str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Text chunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits documents into overlapping chunks for vector indexing.

    Strategy: sentence-aware chunking with a sliding window.
    Overlap between chunks ensures that sentences spanning chunk boundaries
    are retrievable from either chunk — critical for policy documents where
    a rule and its exception may span a page boundary.

    Attributes:
        chunk_size:    Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with inherited metadata.

        Args:
            text:     Full document text.
            metadata: Document-level metadata (filename, doc_type, etc.)
                      inherited by all chunks.

        Returns:
            List of chunk dicts with keys: text, metadata, chunk_index.
        """
        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text.strip())

        # Split on sentence boundaries first, then assemble into chunks
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks    = []
        current   = ""
        idx       = 0

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append({
                        "text":        current,
                        "metadata":    {**metadata, "chunk_index": idx},
                        "chunk_index": idx,
                    })
                    idx += 1
                    # Overlap: keep the tail of the previous chunk
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current       = current[overlap_start:].strip() + " " + sentence
                    current       = current.strip()
                else:
                    current = sentence

        if current:
            chunks.append({
                "text":        current,
                "metadata":    {**metadata, "chunk_index": idx},
                "chunk_index": idx,
            })

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Document loaders
# ─────────────────────────────────────────────────────────────────────────────

class DocumentLoader:
    """Loads documents from disk and returns extracted plain text."""

    @staticmethod
    def load(filepath: str) -> str:
        """
        Load a document and return its plain text.

        Supported formats: .txt, .md, .json, .pdf (via pdfplumber), .docx (via docx2txt)
        Falls back to raw UTF-8 read for unknown extensions.
        """
        path = Path(filepath)
        ext  = path.suffix.lower()

        if ext in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="replace")

        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            # Handle our synthetic HR document JSON format
            if isinstance(data, dict) and "content" in data:
                return data["content"]
            return json.dumps(data, indent=2)

        if ext == ".pdf":
            return DocumentLoader._load_pdf(filepath)

        if ext == ".docx":
            return DocumentLoader._load_docx(filepath)

        # Fallback
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read '%s': %s", filepath, exc)
            return ""

    @staticmethod
    def _load_pdf(filepath: str) -> str:
        """Extract text from PDF using pdfplumber (optional dependency)."""
        try:
            import pdfplumber  # noqa: PLC0415
            text_parts = []
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber not installed — reading PDF as binary text")
            return Path(filepath).read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _load_docx(filepath: str) -> str:
        """Extract text from DOCX using docx2txt (optional dependency)."""
        try:
            import docx2txt  # noqa: PLC0415
            return docx2txt.process(filepath)
        except ImportError:
            logger.warning("docx2txt not installed — reading DOCX as binary text")
            return Path(filepath).read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding engine
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers for local text embedding.

    Falls back to a simple TF-IDF-style bag-of-words embedding when
    sentence-transformers is not installed — so the agent always runs,
    just with lower semantic quality.

    Attributes:
        model_name: Hugging Face model identifier.
        _model:     Loaded SentenceTransformer model (lazy-loaded).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model     = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings and return a list of float vectors."""
        model = self._get_model()
        if model is not None:
            vecs = model.encode(texts, show_progress_bar=False)
            return vecs.tolist()
        return self._tfidf_embed(texts)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer '%s' loaded", self.model_name)
            return self._model
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — using TF-IDF fallback embedding"
            )
            return None

    @staticmethod
    def _tfidf_embed(texts: List[str]) -> List[List[float]]:
        """
        Minimal bag-of-words embedding fallback.

        Produces a 256-dimensional binary term-presence vector using a
        deterministic hash of each word.  Not semantic — but functional
        for basic keyword matching when sentence-transformers is unavailable.
        """
        import hashlib

        DIM = 256
        results = []
        for text in texts:
            vec    = [0.0] * DIM
            words  = re.findall(r"\w+", text.lower())
            for word in set(words):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                vec[idx] = 1.0
            # Normalise
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# In-memory vector store (ChromaDB-compatible interface)
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Lightweight in-memory vector store with cosine similarity search.

    In production this would be backed by ChromaDB, Pinecone, or
    Azure AI Search.  The interface is identical — only the __init__
    and persistence layer change.

    Attributes:
        _documents:   List of stored chunk dicts.
        _embeddings:  Parallel list of embedding vectors.
    """

    def __init__(self) -> None:
        self._documents:  List[Dict[str, Any]] = []
        self._embeddings: List[List[float]]    = []

    def add(
        self,
        documents:  List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add documents and their embeddings to the store."""
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: List[float],
        top_k:           int = 5,
        filter_metadata: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Cosine similarity search over all stored documents.

        Args:
            query_embedding:  Embedded query vector.
            top_k:            Number of results to return.
            filter_metadata:  Optional metadata filter (key-value exact match).

        Returns:
            List of (document_dict, similarity_score) tuples, sorted by score.
        """
        if not self._documents:
            return []

        import math

        def cosine(a: List[float], b: List[float]) -> float:
            dot  = sum(x * y for x, y in zip(a, b))
            norm = (
                math.sqrt(sum(x * x for x in a)) *
                math.sqrt(sum(y * y for y in b))
            )
            return dot / norm if norm > 0 else 0.0

        scored = []
        for doc, emb in zip(self._documents, self._embeddings):
            if filter_metadata:
                match = all(
                    doc.get("metadata", {}).get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            sim = cosine(query_embedding, emb)
            scored.append((doc, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings.clear()

    def save(self, path: str) -> None:
        """Persist the vector store to disk as JSON."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(self._documents, f)
        with open(os.path.join(path, "embeddings.json"), "w") as f:
            json.dump(self._embeddings, f)

    def load(self, path: str) -> bool:
        """Load a persisted vector store from disk."""
        doc_path = os.path.join(path, "documents.json")
        emb_path = os.path.join(path, "embeddings.json")
        if not os.path.isfile(doc_path):
            return False
        try:
            with open(doc_path) as f:
                self._documents = json.load(f)
            with open(emb_path) as f:
                self._embeddings = json.load(f)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load vector store: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Core knowledge base agent
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBaseAgent:
    """
    RAG engine over the HR document corpus.

    Ingests documents at startup or on demand, then answers employee
    questions by retrieving relevant policy chunks and generating a
    grounded answer with citations.

    Attributes:
        chunker:         TextChunker instance.
        embedder:        EmbeddingEngine instance.
        vector_store:    VectorStore instance.
        loader:          DocumentLoader.
        index_path:      Path to persist/load the vector store.
        min_similarity:  Minimum similarity score to consider a chunk relevant.
        top_k:           Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        index_path:     str   = DEFAULT_INDEX_PATH,
        min_similarity: float = 0.25,
        top_k:          int   = 5,
    ) -> None:
        self.chunker        = TextChunker(chunk_size=800, chunk_overlap=150)
        self.embedder       = EmbeddingEngine()
        self.vector_store   = VectorStore()
        self.loader         = DocumentLoader()
        self.index_path     = index_path
        self.min_similarity = min_similarity
        self.top_k          = top_k
        self._doc_registry: Dict[str, IngestResult] = {}

        # Try to load an existing index from disk
        if self.vector_store.load(index_path):
            logger.info(
                "Vector store loaded from '%s'  chunks=%d",
                index_path, self.vector_store.count(),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def ingest_document(
        self,
        filepath: str,
        doc_type: str = "policy",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        """
        Load a document from disk, chunk it, embed the chunks, and add to the
        vector store.

        Args:
            filepath:  Path to the document file.
            doc_type:  Category tag (policy, handbook, faq, benefit, procedure).
            metadata:  Additional metadata to attach to all chunks from this doc.

        Returns:
            IngestResult with chunk count and success status.
        """
        filename = os.path.basename(filepath)
        doc_id   = hashlib.md5(filepath.encode()).hexdigest()[:12]

        logger.info("Ingesting '%s'  type=%s", filename, doc_type)

        # Load text
        text = self.loader.load(filepath)
        if not text.strip():
            return IngestResult(doc_id, filename, 0, doc_type, False, "Empty document")

        # Metadata attached to every chunk from this document
        doc_meta = {
            "doc_id":     doc_id,
            "filename":   filename,
            "doc_type":   doc_type,
            "filepath":   filepath,
            "ingested_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        # Chunk the document
        chunks = self.chunker.chunk(text, doc_meta)
        if not chunks:
            return IngestResult(doc_id, filename, 0, doc_type, False, "No chunks produced")

        # Embed and store
        texts      = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(chunks, embeddings)

        # Persist updated index
        self.vector_store.save(self.index_path)

        result = IngestResult(doc_id, filename, len(chunks), doc_type, True)
        self._doc_registry[doc_id] = result
        logger.info("Ingested '%s'  chunks=%d  doc_id=%s", filename, len(chunks), doc_id)
        return result

    def ingest_directory(
        self,
        directory: str = DEFAULT_CORPUS_PATH,
        doc_type:  str = "policy",
    ) -> List[IngestResult]:
        """Ingest all supported files from a directory."""
        supported = {".txt", ".md", ".pdf", ".docx", ".json"}
        results   = []
        for filepath in sorted(Path(directory).rglob("*")):
            if filepath.suffix.lower() in supported and filepath.is_file():
                results.append(self.ingest_document(str(filepath), doc_type=doc_type))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        question:    str,
        doc_type:    Optional[str] = None,
        top_k:       Optional[int] = None,
    ) -> RAGResult:
        """
        Answer an employee's HR question using retrieval-augmented generation.

        Retrieval:
            1. Embed the question.
            2. Cosine search the vector store for the top-K most similar chunks.
            3. Filter out chunks below the similarity threshold.

        Generation:
            4. Assemble a grounded prompt with the retrieved chunks as context.
            5. Call the Anthropic API to generate a cited answer.
            6. If no relevant chunks found, return a "not found" response.

        Args:
            question:  Employee's natural-language question.
            doc_type:  Optional filter — only search within this document type.
            top_k:     Override the default number of chunks to retrieve.

        Returns:
            RAGResult with answer, citations, and confidence.
        """
        import time
        t0 = time.perf_counter()

        if self.vector_store.count() == 0:
            return RAGResult(
                query          = question,
                answer         = (
                    "The HR knowledge base is empty. "
                    "Please ask your HR team to ingest the policy documents first."
                ),
                citations      = [],
                chunks_used    = [],
                confidence     = 0.0,
                found_relevant = False,
                latency_ms     = 0.0,
            )

        # Embed the query
        query_vec = self.embedder.embed([question])[0]
        k         = top_k or self.top_k

        # Retrieve top-K chunks
        filter_meta = {"doc_type": doc_type} if doc_type else None
        results     = self.vector_store.search(query_vec, top_k=k, filter_metadata=filter_meta)

        # Filter by similarity threshold
        relevant = [(doc, score) for doc, score in results if score >= self.min_similarity]

        if not relevant:
            latency = (time.perf_counter() - t0) * 1000
            return RAGResult(
                query          = question,
                answer         = (
                    "I couldn't find a specific policy covering this question in the "
                    "HR knowledge base. Please contact your HR representative directly "
                    "or submit a ticket to the HR helpdesk."
                ),
                citations      = [],
                chunks_used    = [],
                confidence     = 0.0,
                found_relevant = False,
                latency_ms     = latency,
            )

        # Build context from retrieved chunks
        context_parts = []
        for i, (doc, score) in enumerate(relevant, start=1):
            meta = doc.get("metadata", {})
            context_parts.append(
                f"[Source {i}: {meta.get('filename', 'Unknown')}]\n{doc['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Generate grounded answer via Anthropic API
        answer = self._generate_answer(question, context, relevant)

        # Build citations
        citations = []
        seen_docs = set()
        for doc, score in relevant:
            meta    = doc.get("metadata", {})
            doc_key = meta.get("filename", "") + str(meta.get("chunk_index", ""))
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                citations.append({
                    "filename":    meta.get("filename", "Unknown"),
                    "doc_type":    meta.get("doc_type", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity":  round(score, 3),
                    "excerpt":     doc["text"][:200] + "…" if len(doc["text"]) > 200 else doc["text"],
                })

        confidence = relevant[0][1] if relevant else 0.0
        latency    = (time.perf_counter() - t0) * 1000

        logger.info(
            "Query answered  question='%s...'  chunks=%d  confidence=%.3f  latency=%.1fms",
            question[:50], len(relevant), confidence, latency,
        )

        return RAGResult(
            query          = question,
            answer         = answer,
            citations      = citations,
            chunks_used    = [{"text": d["text"][:300], "score": s} for d, s in relevant],
            confidence     = confidence,
            found_relevant = True,
            latency_ms     = latency,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_answer(
        self,
        question: str,
        context:  str,
        relevant: List[Tuple[Dict, float]],
    ) -> str:
        """
        Generate a grounded answer using the Anthropic API.

        The prompt is structured as a strict RAG prompt:
            - System: "You are an HR assistant. Answer ONLY from the provided context."
            - Context: retrieved policy chunks
            - User: employee's question

        If the Anthropic API is not available, falls back to an extractive
        answer that quotes the most relevant chunk directly.
        """
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic()
            system = (
                "You are a knowledgeable HR assistant for an enterprise organisation. "
                "Answer the employee's question using ONLY the HR policy context provided below. "
                "Be specific, accurate, and cite which policy document each piece of information "
                "comes from (use [Source N] notation). "
                "If the context does not contain enough information to answer fully, say so clearly "
                "and suggest the employee contact HR directly. "
                "Never invent policy details not present in the context. "
                "Be concise but complete — employees need actionable answers."
            )
            prompt = (
                f"HR POLICY CONTEXT:\n\n{context}\n\n"
                f"EMPLOYEE QUESTION: {question}\n\n"
                "Please provide a clear, accurate answer citing the relevant policy sources."
            )

            response = client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 800,
                system     = system,
                messages   = [{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except ImportError:
            logger.warning("anthropic library not installed — using extractive fallback")
            return self._extractive_answer(question, relevant)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed: %s — using extractive fallback", exc)
            return self._extractive_answer(question, relevant)

    @staticmethod
    def _extractive_answer(
        question: str,
        relevant: List[Tuple[Dict, float]],
    ) -> str:
        """
        Extractive fallback: return the most relevant chunk with framing.

        Used when the Anthropic API is unavailable — returns the raw
        policy text rather than generating a synthesised answer.
        """
        if not relevant:
            return "No relevant policy found for this question."

        best_doc, best_score = relevant[0]
        meta     = best_doc.get("metadata", {})
        excerpt  = best_doc["text"][:600]
        filename = meta.get("filename", "HR Policy Document")

        return (
            f"Based on {filename}:\n\n"
            f"{excerpt}\n\n"
            f"(Similarity score: {best_score:.2f} — for a complete answer, "
            f"please consult the full document or contact HR.)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def index_stats(self) -> Dict[str, Any]:
        """Return statistics about the current vector index."""
        return {
            "total_chunks": self.vector_store.count(),
            "documents":    len(self._doc_registry),
            "index_path":   self.index_path,
        }

    def clear_index(self) -> None:
        """Clear all chunks from the vector store (use with caution)."""
        self.vector_store.clear()
        self._doc_registry.clear()
        logger.warning("Vector store cleared")
