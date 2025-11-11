from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import data, retrieve

from pipeline_utils import load_model_and_tokenizer, sample_generations
from uncertainty_estimation import semantic_entropy, sum_eigen

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device)

docs = data.data("wikimedia/wikipedia", 100)
bm25 = retrieve.build_bm25(docs)

question = "Who wrote The Old Man and the Sea?"

# sample n generations with logprobs
generations = sample_generations(
    model=model,
    tokenizer=tokenizer,
    question=question,
    bm25=bm25,
    documents=docs,
    k_ctx=5,   # how many retrieved passages to include
    n=5,       # number of generations
    max_new_tokens=128,
    temperature=0.9,
    top_p=0.95,
    base_seed=0,
)

# print("Generations:", *generations["generated_texts"], sep="\n- ")

# metrics
se = semantic_entropy(generations, question)
su = sum_eigen({"generated_texts": generations["generated_texts"]}, question)

print("\nSemantic Entropy:", se["semantic_entropy"], " (truth_value:", se["truth_value"], ")")
print("Sum Eigen Uncertainty:", su["U_eigv"], " (truth_value:", su["truth_value"], ")")
