# retrievers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Optional, Callable, Sequence, Union, Any

from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Common, tiny interface
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
# BM25 retriever (sparse)
# ---------------------------

class BM25Retriever:
    """
    A thin wrapper around rank_bm25 that implements the Retriever interface.
    Works with list[str] *or* list[dict] (use text_key for dicts).
    """

    def __init__(
        self,
        documents: Sequence[Union[str, dict]],
        *,
        text_key: str = "text",
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        ):
        self._docs = documents
        self._text_key = text_key
        self._tok = tokenizer or (lambda s: s.split())
        self._tokenized_docs = [self._tok(self._get_text(d)) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized_docs)

    def _get_text(self, d: Union[str, dict]) -> str:
        return d.get(self._text_key, "") if isinstance(d, dict) else str(d)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        tokens = self._tok(query)
        scores = self._bm25.get_scores(tokens)  # np.ndarray
        # top indices descending
        idxs = scores.argsort()[-top_k:][::-1]
        out: List[RetrievedChunk] = []
        for i in idxs:
            text = self._get_text(self._docs[i])
            out.append(
                RetrievedChunk(
                    text=text,
                    source_id=str(i),
                    score=float(scores[i]),
                    meta={"doc": self._docs[i]},
                )
            )
        return out


# ---------------------------
# Contriever retriever (dense)
# ---------------------------

class ContrieverRetriever:
    """
    Dense retriever using HF 'facebook/contriever' (or contriever-msmarco).
    - Builds document embeddings once (with batching).
    - Uses FAISS if available (IP index with normalized embeddings) else torch matmul.
    - Compatible with list[str] or list[dict] corpora.
    """

    def __init__(
        self,
        documents: Sequence[Union[str, dict]],
        *,
        model_name: str = "facebook/contriever",
        device: Optional[str] = None,          # "cuda" | "mps" | "cpu" (auto if None)
        text_key: str = "text",
        batch_size: int = 32,
        normalize: bool = True,                # L2 normalize for cosine/IP
        show_progress: bool = True,
    ):
        self.torch = torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModel = AutoModel

        self._docs = documents
        self._text_key = text_key
        self._batch_size = batch_size
        self._normalize = normalize

        # device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        # model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        # encode docs
        self._doc_texts = [self._get_text(d) for d in self._docs]
        self._embeds = self._encode_texts(self._doc_texts, show_progress=show_progress)  # (N, D)

        # FAISS index (optional)
        self._faiss = None
        try:
            import faiss  # type: ignore
            self._faiss = faiss
        except Exception:
            self._faiss = None

        if self._faiss is not None:
            # Use inner product; with normalized vectors this is cosine similarity
            dim = self._embeds.shape[1]
            if self._normalize:
                index = self._faiss.IndexFlatIP(dim)
            else:
                # If not normalizing, cosine ≠ IP; IP is fine but scales differ.
                index = self._faiss.IndexFlatIP(dim)
            index.add(self._embeds.astype("float32"))
            self._faiss_index = index
        else:
            self._faiss_index = None  # fallback to torch matmul

    def _get_text(self, d: Union[str, dict]) -> str:
        return d.get(self._text_key, "") if isinstance(d, dict) else str(d)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        # standard mean pooling
        import torch
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode_texts(self, texts: List[str], show_progress: bool = True):
        import numpy as np
        from math import ceil

        bs = self._batch_size
        n = len(texts)
        n_steps = ceil(n / bs)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                rng = tqdm(range(n_steps), desc="Encoding docs")
            except Exception:
                rng = range(n_steps)
        else:
            rng = range(n_steps)

        all_vecs = []
        with self.torch.no_grad():
            for step in rng:
                chunk = texts[step * bs : (step + 1) * bs]
                toks = self._tokenizer(
                    chunk, padding=True, truncation=True, return_tensors="pt"
                ).to(self._device)
                out = self._model(**toks)
                vecs = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
                if self._normalize:
                    vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-12)
                all_vecs.append(vecs.cpu())
        mat = self.torch.cat(all_vecs, dim=0).numpy().astype("float32")  # (N, D)
        return mat

    def _encode_query(self, query: str):
        with self.torch.no_grad():
            toks = self._tokenizer(query, return_tensors="pt", truncation=True).to(self._device)
            out = self._model(**toks)
            vec = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
            if self._normalize:
                vec = vec / (vec.norm(dim=1, keepdim=True) + 1e-12)
        return vec.squeeze(0).detach().cpu().numpy().astype("float32")  # (D,)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        import numpy as np

        q = self._encode_query(query)  # (D,)
        if self._faiss_index is not None:
            D, I = self._faiss_index.search(q.reshape(1, -1), top_k)  # distances, indices
            idxs = I[0].tolist()
            scores = D[0].tolist()
        else:
            # torch matmul fallback
            sims = (self._embeds @ q.reshape(-1, 1)).squeeze(-1)  # (N,)
            idxs = sims.argsort()[-top_k:][::-1].tolist()
            scores = sims[idxs].tolist()

        out: List[RetrievedChunk] = []
        for i, s in zip(idxs, scores):
            text = self._doc_texts[i]
            out.append(
                RetrievedChunk(
                    text=text,
                    source_id=str(i),
                    score=float(s),
                    meta={"doc": self._docs[i]},
                )
            )
        return out