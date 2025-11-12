import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List
from tqdm.auto import tqdm

import torch, random

class LLM:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache_dict = {}

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: int | None = None,
        return_logprobs: bool = False,
    ):
        """
        Generate text continuation from a prompt using a local LLM.
        Returns:
            - continuation (str)
            - token_logprobs (list[float]) if return_logprobs=True
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        gen_ids = gen_out.sequences[0][input_len:]
        continuation = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        if not return_logprobs:
            return continuation, None

        # per-token logprobs
        token_logprobs = []
        for step_scores, tok_id in zip(gen_out.scores, gen_ids):
            logprobs = torch.log_softmax(step_scores, dim=-1)
            token_logprobs.append(logprobs[..., tok_id].item())

        return continuation, token_logprobs

    def save_cache(self):
        pass

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)

    llm = LLM(model, tokenizer, device)
    return llm


def _text_from_doc(d: Any) -> str:
    # Supports plain strings or dicts with a "text" field
    return d.get("text", d) if isinstance(d, dict) else str(d)


def _text_from_hit(hit: Any) -> str:
    """
    Works with either:
      - RetrievedChunk(text=..., meta={"doc": original_doc})
      - Plain string doc
      - Dict doc with "text" field
    """
    if hasattr(hit, "text"):
        return str(getattr(hit, "text"))
    return _text_from_doc(hit)


def build_prompt(question: str, retriever, k: int = 5) -> str:
    """
    Generic prompt builder that uses any retriever implementing:
        retriever.search(query: str, top_k: int) -> List[RetrievedChunk or doc]
    """
    hits: List[Any] = retriever.search(question, top_k=k)
    context_blocks = []
    for i, h in enumerate(hits, 1):
        context_blocks.append(f"[{i}] {_text_from_hit(h)}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"

    return (
        "You are a helpful assistant. Use the context to answer concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def sample_generations(
    llm,
    question: str,
    retriever,
    k_ctx: int,
    n: int,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
    base_seed: int = 0,
):
    prompt = build_prompt(question, retriever, k=k_ctx)
    generated_texts = []
    logprobs_per_gen = []

    for i in tqdm(range(n), desc="generations"):
        seed = None if base_seed is None else (base_seed + i)
        text, token_logprobs = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            return_logprobs=True,
        )

        generated_texts.append(text)
        logprobs_per_gen.append(token_logprobs)

    return {
        "generated_texts": generated_texts,
        "logprobs": logprobs_per_gen,
    }