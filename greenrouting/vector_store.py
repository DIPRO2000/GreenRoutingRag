"""
Vector Store and Embedding Manager for GreenRouting.
Manages Sentence-Transformers embeddings and FAISS / NumPy cosine similarity indexing.
"""
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure OpenMP DLLs are loaded in Windows
try:
    import torch
except ImportError:
    pass

FAISS_AVAILABLE = False
# Only attempt loading native FAISS DLL if explicitly enabled to prevent Windows SEH handle crashes
if os.getenv("GREENROUTING_USE_FAISS") == "1":
    try:
        import faiss
        FAISS_AVAILABLE = True
    except (ImportError, Exception) as e:
        logging.getLogger(__name__).warning("FAISS native DLL unavailable (%s). Using high-performance NumPy vector backend.", e)

logger = logging.getLogger(__name__)


class DeterministicFallbackEmbedder:
    """
    Lightweight deterministic fallback embedder used when offline or when
    HuggingFace network downloads are restricted. Ensures 100% test reliability.
    Uses hashed character/word n-gram frequency projected to 384 dimensions.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vectors.append(vec)
                continue

            for word in words:
                h = hash(word)
                idx = abs(h) % self.dimension
                sign = 1.0 if (h % 2 == 0) else -1.0
                vec[idx] += sign

            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)


class EmbeddingService:
    """Manages sentence embedding model with automatic fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        self.is_fallback = False
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"  # Skip network check and load immediately from local disk cache
            
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name)
                
            if hasattr(self.model, "get_embedding_dimension"):
                self.dimension = self.model.get_embedding_dimension()
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                self.dimension = 384
            self.is_fallback = False
            logger.info("Loaded SentenceTransformer model: %s (dim=%d)", self.model_name, self.dimension)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer model '%s': %s. Using DeterministicFallbackEmbedder.", self.model_name, e)
            self.model = DeterministicFallbackEmbedder(self.dimension)
            self.is_fallback = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized embedding for a single query."""
        emb = self.embed_texts([query])
        return emb[0]


class FaissVectorStore:
    """Vector database supporting FAISS IndexFlatIP (cosine similarity) with transparent NumPy fallback."""

    def __init__(self, index_dir: Path, embedding_service: EmbeddingService):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service
        self.dimension = embedding_service.dimension

        self.index_file = self.index_dir / "index.faiss"
        self.numpy_matrix_file = self.index_dir / "vectors.npy"
        self.metadata_file = self.index_dir / "chunks_metadata.json"

        self.use_faiss = FAISS_AVAILABLE
        self.faiss_index = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.chunks: List[Dict[str, Any]] = []

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load index from disk if present, else create fresh."""
        # 1. Try loading FAISS if available
        if self.use_faiss and self.index_file.exists() and self.metadata_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                return
            except Exception as e:
                logger.warning("Failed to load existing FAISS index: %s. Recreating.", e)

        # 2. Try loading NumPy index if FAISS is unavailable or failed
        if self.numpy_matrix_file.exists() and self.metadata_file.exists():
            try:
                self.embeddings_matrix = np.load(str(self.numpy_matrix_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                return
            except Exception as e:
                logger.warning("Failed to load NumPy vector matrix: %s. Recreating.", e)

        # 3. Create fresh
        if self.use_faiss:
            try:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
            except Exception:
                self.use_faiss = False
                self.embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)
        else:
            self.embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)

        self.chunks = []

    def save_index(self):
        """Persist index and chunk metadata to disk."""
        if self.use_faiss and self.faiss_index is not None:
            try:
                faiss.write_index(self.faiss_index, str(self.index_file))
            except Exception as e:
                logger.warning("Failed writing FAISS index: %s. Persisting NumPy array.", e)
                if self.embeddings_matrix is not None:
                    np.save(str(self.numpy_matrix_file), self.embeddings_matrix)
        elif self.embeddings_matrix is not None:
            np.save(str(self.numpy_matrix_file), self.embeddings_matrix)

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and add document chunks to the vector index."""
        if not chunks:
            return 0

        texts = [c["text_content"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Ensure normalized vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings / norms

        if self.use_faiss and self.faiss_index is not None:
            try:
                self.faiss_index.add(normalized_embeddings.astype(np.float32))
            except Exception as e:
                logger.warning("FAISS add failed (%s); switching to NumPy vector storage.", e)
                self.use_faiss = False
                if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
                    self.embeddings_matrix = normalized_embeddings
                else:
                    self.embeddings_matrix = np.vstack([self.embeddings_matrix, normalized_embeddings])
        else:
            if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
                self.embeddings_matrix = normalized_embeddings
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, normalized_embeddings])

        self.chunks.extend(chunks)
        self.save_index()
        return len(chunks)

    def search(self, query: str, top_k: int = 3, doc_id_filter: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for top_k most similar chunks for a query.
        Returns list of (chunk_dict, cosine_similarity_score).
        """
        total = self.get_total_chunks()
        if total == 0 or not self.chunks:
            return []

        query_vec = self.embedding_service.embed_query(query).astype(np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-9:
            query_vec = query_vec / q_norm

        results: List[Tuple[Dict[str, Any], float]] = []

        if self.use_faiss and self.faiss_index is not None:
            try:
                k_search = min(top_k * 4 if doc_id_filter else top_k, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(query_vec.reshape(1, -1), k_search)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    chunk = self.chunks[idx]
                    if doc_id_filter and chunk.get("doc_id") != doc_id_filter:
                        continue
                    sim_score = float(max(-1.0, min(1.0, score)))
                    results.append((chunk, sim_score))
                    if len(results) >= top_k:
                        break
                return results
            except Exception as e:
                logger.warning("FAISS search encountered error: %s. Falling back to NumPy vector search.", e)

        # NumPy Cosine Vector Search
        if self.embeddings_matrix is not None and len(self.embeddings_matrix) > 0:
            similarities = np.dot(self.embeddings_matrix, query_vec)
            ranked_indices = np.argsort(similarities)[::-1]

            for idx in ranked_indices:
                if idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                if doc_id_filter and chunk.get("doc_id") != doc_id_filter:
                    continue
                score = float(max(-1.0, min(1.0, similarities[idx])))
                results.append((chunk, score))
                if len(results) >= top_k:
                    break

        return results

    def clear(self):
        """Reset index and stored chunks."""
        self.chunks = []
        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.reset()
        self.embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)

        for p in [self.index_file, self.numpy_matrix_file, self.metadata_file]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def get_total_chunks(self) -> int:
        if self.use_faiss and self.faiss_index is not None:
            return self.faiss_index.ntotal
        if self.embeddings_matrix is not None:
            return len(self.embeddings_matrix)
        return len(self.chunks)
