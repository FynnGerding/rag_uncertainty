import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, List
from tqdm.auto import tqdm


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
        torch_dtype=dtype,
    ).to(device)
    return tokenizer, model


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

@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.95,
    seed: int | None = None,
):
    """
    Returns: text, token_logprobs (list[float])
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    gen_out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    gen_ids = gen_out.sequences[0][input_len:]

    # per-token logprobs
    token_logprobs = []
    for step_scores, tok_id in zip(gen_out.scores, gen_ids):
        logprobs = torch.log_softmax(step_scores, dim=-1)
        token_logprobs.append(logprobs[..., tok_id].item())

    continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return continuation.strip(), token_logprobs


def sample_generations(
    model,
    tokenizer,
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
        text, token_logprobs = generate_once(
            model,
            tokenizer,
            prompt,
            device=model.device.type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        generated_texts.append(text)
        logprobs_per_gen.append(token_logprobs)

    return {
        "generated_texts": generated_texts,
        "logprobs": logprobs_per_gen,
    }