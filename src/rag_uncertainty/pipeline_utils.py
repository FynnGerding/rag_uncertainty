import random
import torch
import outlines
from outlines import models, generate, samplers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List
from tqdm.auto import tqdm
from typing import Optional, Union
from pydantic import BaseModel

from rag_uncertainty.retrievers import RetrievedChunk

class LLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        
        # Load underlying HF components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        
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
        constraint: Optional[Union[str, type, BaseModel]] = None,
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

        if isinstance(constraint, type) and issubclass(constraint, BaseModel):
            continuation_str = continuation.model_dump_json()
        else:
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
            gen = generate.regex(self.model, constraint, sampler=sampler)
            
        elif isinstance(constraint, type) and issubclass(constraint, BaseModel):
            gen = generate.json(self.model, constraint, sampler=sampler)
            
        elif constraint in [int, float, bool]:
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

def _qwen_prompt(system: str, user: str) -> str:
    """Helper to format prompts for Qwen2.5 Instruct model."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_rag_prompt(question: str, retriever, k: int = 5) -> str:
    """
    Constructs a Qwen 2.5 formatted RAG prompt.
    """
    # Search
    hits: List[RetrievedChunk] = retriever.search(question, top_k=k)
    
    # Construct context
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        # Access the .text attribute directly from the dataclass
        content = hit.text.strip()
        context_blocks.append(f"Document [{i}]:\n{content}")
    
    context_str = "\n\n".join(context_blocks) if context_blocks else "(No relevant context found)"

    # Prompt
    system_msg = (
        "You are an expert research writer. Your task is to write a comprehensive, "
        "detailed answer to the user's question using the provided documents.\n"
        "Guidelines:\n"
        "1. Synthesize information from multiple context blocks into a coherent narrative.\n"
        "2. Prioritize length and detail. Explain the 'who, what, where, when, and why' based on the text.\n"
        "3. If the context is partial or fragmented, construct the best possible answer using the available "
        "evidence, rather than refusing to answer. Connect the dots logically.\n"
        "4. Strict Grounding: Do not hallucinate external details. If a specific detail is missing, "
        "omit it or state that the documents do not specify it, but continue the narrative."
    )
    
    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )

    return _qwen_prompt(system_msg, user_msg), hits

def sample_generations(
    llm: LLM,
    question: str,
    retriever,
    k_ctx: int = 5,
    n: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    ):
    """
    Generates RAG answers using the LLM wrapper and Qwen formatting.
    """
    prompt, hits = build_rag_prompt(question, retriever, k=k_ctx)
    
    generated_texts = []
    logprobs_per_gen = []

    for i in tqdm(range(n), desc="Generations"):
        
        text, token_logprobs = llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=None,
            return_logprobs=True,
        )

        generated_texts.append(text)
        logprobs_per_gen.append(token_logprobs)

    return {
        "prompt_used": prompt,
        "generated_texts": generated_texts,
        "logprobs": logprobs_per_gen,
        "context": hits,
    }
