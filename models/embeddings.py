"""
NeoStats AI Chatbot — Embedding Models Module
Handles document vectorization using SentenceTransformers + FAISS.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Lightweight embedding model wrapper using SentenceTransformers.
    Encodes text into dense vectors for semantic similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load()
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # L2-normalize for cosine similarity
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding encode error: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string and return 1-D vector."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()


class FAISSVectorStore:
    """
    FAISS-backed vector store for fast nearest-neighbour retrieval.
    Stores both the FAISS index and corresponding text metadata.
    """

    def __init__(self, embedding_model: EmbeddingModel, store_path: str = "data/vector_store"):
        self.embedding_model = embedding_model
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index = None
        self.metadata: list[dict] = []   # [{text, source, chunk_id, ...}, ...]
        self._index_path = self.store_path / "index.faiss"
        self._meta_path = self.store_path / "metadata.pkl"

    def _init_index(self):
        """Initialize empty FAISS index."""
        try:
            import faiss
            dim = self.embedding_model.dimension
            self.index = faiss.IndexFlatIP(dim)   # Inner product (cosine since normalized)
            logger.info(f"Initialized FAISS index with dimension {dim}")
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    def add_documents(self, chunks: list[dict]) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of dicts with keys: text, source, chunk_id, metadata

        Returns:
            Number of chunks added
        """
        try:
            if not chunks:
                return 0

            texts = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(texts)

            if self.index is None:
                self._init_index()

            self.index.add(embeddings.astype("float32"))
            self.metadata.extend(chunks)

            # Persist to disk
            self._save()
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def search(self, query: str, top_k: int = 4, threshold: float = 0.25) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: Search query string
            top_k: Max number of results
            threshold: Min cosine similarity to include in results

        Returns:
            List of matching chunk dicts with added 'score' key
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            query_vec = self.embedding_model.encode_single(query).reshape(1, -1).astype("float32")
            scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < threshold:
                    continue
                chunk = dict(self.metadata[idx])
                chunk["score"] = float(score)
                results.append(chunk)

            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _save(self):
        """Persist index and metadata to disk."""
        try:
            import faiss
            faiss.write_index(self.index, str(self._index_path))
            with open(self._meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def load(self) -> bool:
        """Load persisted index and metadata from disk."""
        try:
            if not self._index_path.exists() or not self._meta_path.exists():
                return False
            import faiss
            self.index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded vector store: {self.index.ntotal} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def clear(self):
        """Remove all documents from the vector store."""
        self._init_index()
        self.metadata = []
        if self._index_path.exists():
            self._index_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        logger.info("Vector store cleared")

    @property
    def total_chunks(self) -> int:
        """Return total number of indexed chunks."""
        return len(self.metadata)

    @property
    def sources(self) -> list[str]:
        """Return unique source document names."""
        return list({m.get("source", "unknown") for m in self.metadata})
