# pipeline.py
import torch
import data
import retrievers
from pipeline_utils import load_model_and_tokenizer, sample_generations
from uncertainty_estimation import semantic_entropy, sum_eigen

# pick device & load model
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device)

# load docs (strings or dicts with "text")
docs = data.data("wikimedia/wikipedia", 100)

# choose any retriever that implements .search(query, top_k)
# Option A: sparse BM25
retriever = retrievers.BM25Retriever(docs)

# Option B (swap to dense):
# retriever = retrievers.ContrieverRetriever(docs, model_name="facebook/contriever-msmarco", device=device)

question = "Who wrote The Old Man and the Sea?"

# sample n generations with logprobs
generations = sample_generations(
    model=model,
    tokenizer=tokenizer,
    question=question,
    retriever=retriever,
    k_ctx=5,
    n=5,
    max_new_tokens=128,
    temperature=0.9,
    top_p=0.95,
    base_seed=0,
)

# metrics
se = semantic_entropy(generations, question)
su = sum_eigen({"generated_texts": generations["generated_texts"]}, question)

print("\nSemantic Entropy:", se["semantic_entropy"], "(truth_value:", se["truth_value"], ")")
print("Sum Eigen Uncertainty:", su["U_eigv"], "(truth_value:", su["truth_value"], ")")
