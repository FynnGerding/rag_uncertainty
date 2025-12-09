from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Optional, Callable, Sequence, Union
import logging
import os
import bm25s

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------
# Common Interface
# ---------------------------

@dataclass
class RetrievedChunk:
    text: str
    source_id: Optional[str] = None        # index or doc id
    score: Optional[float] = None
    meta: Optional[dict] = None            # e.g. full doc, title, span, etc.


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Return top_k chunks for a query."""
        ...


# ---------------------------
# BM25 Retriever
# ---------------------------

class BM25Retriever:
    """
    Sparse retriever using 'bm25s'.
    Capable of indexing millions of documents in seconds/minutes.
    
    Args:
        documents: List of strings or dicts.
        text_key: Key to extract text if input is dicts.
        save_dir: (Optional) If provided, saves the index here to avoid re-indexing.
    """

    def __init__(
        self,
        documents: Sequence[Union[str, dict]],
        *,
        text_key: str = "text",
        save_dir: Optional[str] = "bm25_index_cache",
        load_if_exists: bool = True
    ):
        
        self._docs = documents
        self._text_key = text_key
        
        # 1. Attempt to Load Cached Index
        if load_if_exists and save_dir and os.path.exists(save_dir):
            try:
                logger.info(f"Loading BM25 index from {save_dir}...")
                self._retriever = bm25s.BM25.load(save_dir, load_corpus=False)
                logger.info("Index loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Re-indexing.")

        # 2. Extract Text
        logger.info(f"Extracting text from {len(documents)} documents...")
        corpus_texts = [self._get_text(d) for d in documents]

        # 3. Tokenize & Index
        logger.info("Tokenizing corpus (this may take a moment)...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=True)

        logger.info("Building BM25 index...")
        self._retriever = bm25s.BM25(corpus=None)
        self._retriever.index(corpus_tokens, show_progress=True)
        
        # 4. Save Index
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self._retriever.save(save_dir)
                logger.info(f"Saved BM25 index to {save_dir}")
            except Exception as e:
                logger.warning(f"Could not save index: {e}")

    def _get_text(self, d: Union[str, dict]) -> str:
        return d.get(self._text_key, "") if isinstance(d, dict) else str(d)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        import bm25s
        
        query_tokens = bm25s.tokenize(query, stopwords="en")
        
        docs, scores = self._retriever.retrieve(
            query_tokens, 
            k=top_k, 
            corpus=self._docs
        )
        
        doc_list = docs[0]
        score_list = scores[0]

        out: List[RetrievedChunk] = []
        for i, doc_item in enumerate(doc_list):
            text = self._get_text(doc_item)
            
            out.append(
                RetrievedChunk(
                    text=text,
                    source_id=None,
                    score=float(score_list[i]),
                    meta={"doc": doc_item},
                )
            )
        return out