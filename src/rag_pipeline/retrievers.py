from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Optional, Callable, Sequence, Union
import logging
import os
import json
import glob
from pathlib import Path
import bm25s

# Configure logging
logger = logging.getLogger(__name__)
DEFAULT_WIKI_DATASET = "wikimedia/wikipedia"
DEFAULT_DATA_CACHE_DIR = "data"
DEFAULT_BM25_CACHE_DIR = "bm25_index_cache"

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
        documents: Sequence[Union[str, dict]] | None,
        *,
        text_key: str = "text",
        save_dir: Optional[str] = "bm25_index_cache",
        load_if_exists: bool = True
    ):
        self._docs = documents
        self._text_key = text_key
        self._save_dir = save_dir

        # 1. Attempt to Load Cached Index
        if load_if_exists and save_dir and os.path.exists(save_dir):
            try:
                logger.info(f"Loading BM25 index from {save_dir}...")
                self._retriever = bm25s.BM25.load(save_dir, load_corpus=False)
                if self._docs is None:
                    cached_docs = self._load_corpus_from_cache(save_dir)
                    if cached_docs is None:
                        raise ValueError("Corpus cache missing; provide documents to align with BM25 index.")
                    self._docs = cached_docs
                else:
                    corpus_cache = self._corpus_cache_path(save_dir)
                    if not os.path.exists(corpus_cache):
                        self._save_corpus_to_cache(save_dir, [self._get_text(d) for d in self._docs])
                logger.info("Index loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Re-indexing.")

        # 2. Extract Text
        if documents is None:
            raise ValueError("documents must be provided when no cached index is available.")
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
                self._save_corpus_to_cache(save_dir, corpus_texts)
                logger.info(f"Saved BM25 index to {save_dir}")
            except Exception as e:
                logger.warning(f"Could not save index: {e}")

    def _get_text(self, d: Union[str, dict]) -> str:
        return d.get(self._text_key, "") if isinstance(d, dict) else str(d)

    def _corpus_cache_path(self, save_dir: str) -> str:
        return os.path.join(save_dir, "corpus.jsonl")

    def _save_corpus_to_cache(self, save_dir: str, corpus_texts: Sequence[str]) -> None:
        try:
            path = self._corpus_cache_path(save_dir)
            with open(path, "w", encoding="utf-8") as f:
                for text in corpus_texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            logger.info(f"Saved corpus cache to {path}")
        except Exception as e:
            logger.warning(f"Could not save corpus cache: {e}")

    def _load_corpus_from_cache(self, save_dir: str) -> Optional[List[str]]:
        path = self._corpus_cache_path(save_dir)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                corpus = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        corpus.append(item.get("text", line))
                    except json.JSONDecodeError:
                        corpus.append(line)
                return corpus
        except Exception as e:
            logger.warning(f"Could not load corpus cache: {e}")
            return None

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

# Helpers for cached loading/building

def _decode_jsonl_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                texts.append(item.get("text", line))
            except json.JSONDecodeError:
                texts.append(line)
    return texts


def _load_corpus_cache(data_cache_dir: str = DEFAULT_DATA_CACHE_DIR) -> Optional[List[str]]:
    """
    Try to load a cached corpus from the data directory by taking the first *.json* file found.
    """
    base = Path(data_cache_dir)
    json_files = sorted(p for p in base.glob("*.json*") if p.is_file())
    if not json_files:
        return None
    path = json_files[0]
    try:
        return _decode_jsonl_texts(path)
    except Exception as e:
        logger.warning(f"Failed to load cached corpus {path}: {e}")
        return None


def _load_dataset_cache(dataset_name: str, data_cache_dir: str = DEFAULT_DATA_CACHE_DIR) -> Optional[List[str]]:
    pattern = Path(data_cache_dir) / f"{dataset_name.replace('/', '_')}_*.jsonl"
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        return None
    path = Path(matches[0])
    try:
        return _decode_jsonl_texts(path)
    except Exception as e:
        logger.warning(f"Failed to load cached dataset chunks {path}: {e}")
        return None


def build_wikipedia_retriever(
    *,
    data_loader: Optional[Callable[[str], List[str]]] = None,
    cache_dir: str = DEFAULT_BM25_CACHE_DIR,
    data_cache_dir: str = DEFAULT_DATA_CACHE_DIR,
) -> BM25Retriever:
    """
    Build a BM25 retriever for Wikipedia chunks, preferring cached index/corpus, then cached dataset, otherwise downloading via data_loader.
    """
    dataset_name = DEFAULT_WIKI_DATASET
    cache_path = Path(cache_dir)
    corpus_cache = cache_path / "corpus.jsonl"

    # 1) If both index and corpus cache exist, load immediately
    if cache_path.exists() and corpus_cache.exists():
        try:
            return BM25Retriever(None, save_dir=cache_dir, load_if_exists=True)
        except Exception as e:
            logger.info(f"Cached index + corpus failed to load, falling back to rebuild: {e}")

    # 2) Try to recover docs from corpus cache or dataset cache
    docs = _load_corpus_cache(data_cache_dir=data_cache_dir)
    if docs is None:
        docs = _load_dataset_cache(dataset_name, data_cache_dir=data_cache_dir)

    # 3) If still none, fetch via loader
    if docs is None:
        if data_loader is None:
            try:
                import data as data_module
                data_loader = data_module.data  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError("No data_loader provided and failed to import data.data") from e
        logger.info(f"Downloading/preparing dataset {dataset_name}...")
        docs = data_loader(dataset_name)

    # 4) Build retriever; if index exists without corpus, this will load the index and write corpus cache
    return BM25Retriever(docs, save_dir=cache_dir, load_if_exists=True)
