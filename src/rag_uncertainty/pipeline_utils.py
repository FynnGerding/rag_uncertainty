import random
import torch
import outlines
from outlines import models, generate, samplers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List
from tqdm.auto import tqdm
from typing import Optional, Union

class LLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        
        # Load underlying HF components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Wrap for Outlines
        self.model = models.Transformers(self.hf_model, self.tokenizer)
        
        # Cache now keys on (constraint, temperature) because the sampler is baked in
        self.generator_cache = {} 

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0, 
        seed: Optional[int] = None,
        constraint: Optional[Union[str, type]] = None, 
        return_logprobs: bool = False,
    ):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)

        # 1. Get the generator (baked with the specific constraint AND temperature)
        generator = self._get_generator(constraint, temperature)

        # 2. Generate
        continuation = generator(prompt, max_tokens=max_new_tokens)

        continuation_str = str(continuation)

        if not return_logprobs:
            return continuation, None

        # 3. Score sequence for logprobs
        token_logprobs = self._score_sequence(prompt, continuation_str)
        
        return continuation, token_logprobs

    def _get_generator(self, constraint: Union[str, type, None], temperature: float):
        """
        Retrieves a generator from cache or compiles a new one.
        The cache key is (constraint, temperature) because the sampler is immutable.
        """
        cache_key = (constraint, temperature)
        
        if cache_key in self.generator_cache:
            return self.generator_cache[cache_key]

        if temperature == 0.0:
            sampler = samplers.greedy()
        else:
            sampler = samplers.multinomial(temperature=temperature)

        # Compile new generator
        if constraint is None:
            gen = generate.text(self.model, sampler=sampler)
        elif isinstance(constraint, str):
            # Assume Regex
            gen = generate.regex(self.model, constraint, sampler=sampler)
        elif constraint in [int, float, bool]:
            # Native Types
            gen = generate.format(self.model, constraint, sampler=sampler)
        else:
            raise ValueError(f"Unsupported constraint type: {constraint}")

        self.generator_cache[cache_key] = gen
        return gen

    @torch.no_grad()
    def _score_sequence(self, prompt: str, continuation: str) -> List[float]:
        """
        Re-runs a forward pass to extract logprobs for the generated sequence.
        """
        full_text = prompt + continuation
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        
        outputs = self.hf_model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        gathered_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
        
        if prompt_len - 1 < gathered_logprobs.shape[1]:
            continuation_logprobs = gathered_logprobs[0, prompt_len-1:].tolist()
        else:
            continuation_logprobs = []
            
        return continuation_logprobs

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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
            seed=seed,
            return_logprobs=True,
        )

        generated_texts.append(text)
        logprobs_per_gen.append(token_logprobs)

    return {
        "generated_texts": generated_texts,
        "logprobs": logprobs_per_gen,
    }