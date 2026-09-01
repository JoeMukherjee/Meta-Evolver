"""Vector reasoning memory bank with MMR & Cosine retrieval."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

class ReasoningMemoryBank:
    """Persistent memory store with diversity-aware Maximal Marginal Relevance (MMR)."""
    def __init__(self, items: Optional[List[Dict[str, Any]]] = None) -> None:
        self.items: List[Dict[str, Any]] = items or []

    @classmethod
    def load_jsonl(cls, path: str | Path) -> ReasoningMemoryBank:
        p = Path(path)
        items = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return cls(items)

    def save_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(item) + "\n")

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        mode: str = "mmr",
        mmr_lambda: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if not self.items:
            return []
        
        # If no embeddings, return top_k plain items
        has_emb = query_embedding is not None and all("embedding" in it and it["embedding"] for it in self.items)
        if not has_emb:
            return self.items[:top_k]
        
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
            
        doc_vecs = np.array([it["embedding"] for it in self.items], dtype=np.float32)
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vecs = doc_vecs / norms
        
        sims = np.dot(doc_vecs, q_vec)
        
        if mode == "cosine":
            top_indices = np.argsort(-sims)[:top_k]
            return [self.items[i] for i in top_indices]
            
        # MMR Mode
        selected: List[int] = []
        candidates = list(range(len(self.items)))
        
        while len(selected) < min(top_k, len(self.items)):
            best_score = -float("inf")
            best_idx = None
            
            for c in candidates:
                relevance = sims[c]
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(np.dot(doc_vecs[c], doc_vecs[s]) for s in selected)
                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = c
                    
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)
            
        return [self.items[i] for i in selected]
