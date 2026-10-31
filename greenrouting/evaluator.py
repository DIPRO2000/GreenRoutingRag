"""
Evaluation Metrics for GreenRouting.
Computes groundedness, context relevance, answer relevance, and composite quality scores
for RAG outputs without relying on external non-deterministic cloud APIs.
"""
import re
from typing import Dict, Any, List, Optional
import numpy as np
from greenrouting.vector_store import EmbeddingService


class RagEvaluator:
    """Evaluates RAG generation quality across faithfulness, relevance, and grounding."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def evaluate(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute standard RAG quality metrics:
        1. Groundedness / Faithfulness: Proportion of answer vocabulary supported by retrieved context.
        2. Context Relevance: Alignment between query and retrieved chunks.
        3. Answer Relevance: Semantic similarity between query and generated answer.
        4. Composite Score: Weighted geometric / harmonic blend of quality metrics.
        """
        clean_ans = answer.strip()
        clean_q = query.strip()
        context_text = " ".join(c.get("text_content", "") for c in retrieved_chunks)

        # 1. Groundedness (Lexical & Statement overlap)
        ans_tokens = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", clean_ans)
                      if w.lower() not in {"the", "and", "that", "this", "with", "from", "based", "according"}]
        context_lower = context_text.lower()
        if not ans_tokens:
            groundedness = 0.0
        else:
            supported = sum(1 for tok in ans_tokens if tok in context_lower)
            groundedness = min(1.0, supported / len(ans_tokens))

        # 2. Context Relevance (Cosine similarity between query and retrieved chunks)
        if retrieved_chunks and clean_q:
            q_emb = self.embedding_service.embed_query(clean_q)
            ctx_snippet = context_text[:1000]
            c_emb = self.embedding_service.embed_query(ctx_snippet)
            q_norm = np.linalg.norm(q_emb)
            c_norm = np.linalg.norm(c_emb)
            if q_norm > 1e-9 and c_norm > 1e-9:
                context_relevance = float(np.dot(q_emb / q_norm, c_emb / c_norm))
                context_relevance = max(0.0, min(1.0, context_relevance))
            else:
                context_relevance = 0.5
        else:
            context_relevance = 0.0

        # 3. Answer Relevance (Semantic similarity between question and answer)
        if clean_q and clean_ans:
            q_emb = self.embedding_service.embed_query(clean_q)
            a_emb = self.embedding_service.embed_query(clean_ans[:500])
            q_norm = np.linalg.norm(q_emb)
            a_norm = np.linalg.norm(a_emb)
            if q_norm > 1e-9 and a_norm > 1e-9:
                ans_relevance = float(np.dot(q_emb / q_norm, a_emb / a_norm))
                ans_relevance = max(0.0, min(1.0, ans_relevance))
            else:
                ans_relevance = 0.5
        else:
            ans_relevance = 0.0

        # 4. Composite Quality Score
        composite_score = (0.50 * groundedness) + (0.25 * context_relevance) + (0.25 * ans_relevance)

        return {
            "groundedness": round(groundedness, 4),
            "context_relevance": round(context_relevance, 4),
            "answer_relevance": round(ans_relevance, 4),
            "composite_score": round(composite_score, 4)
        }
