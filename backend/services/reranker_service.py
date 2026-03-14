from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

# CrossEncoder lives in sentence-transformers
from sentence_transformers import CrossEncoder


@dataclass
class RerankItem:
    chunk_id: str
    document_name: str
    classification: str
    content: str
    base_similarity: float  # similarity_score from vector search


class RerankerService:
    """
    Cross-encoder reranker:
    - Much better ranking than vector distance alone
    - No LLM calls
    - Use: rerank(question, candidates, top_n)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # lighter and Faster: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(
        self,
        question: str,
        candidates: List[RerankItem],
        top_n: int = 8,
        max_chars_per_chunk: int = 2500,
    ) -> List[Tuple[RerankItem, float]]:
        if not candidates:
            return []

        # Prepare pairs for scoring
        pairs = []
        trimmed_items: List[RerankItem] = []
        for c in candidates:
            trimmed = c.content[:max_chars_per_chunk]
            trimmed_item = RerankItem(
                chunk_id=c.chunk_id,
                document_name=c.document_name,
                classification=c.classification,
                content=trimmed,
                base_similarity=c.base_similarity,
            )
            trimmed_items.append(trimmed_item)
            pairs.append((question, trimmed))

        # Cross-encoder scores (higher = more relevant)
        scores = self.model.predict(pairs)

        scored = list(zip(trimmed_items, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[: max(1, top_n)]


# singleton
reranker_service = RerankerService()
