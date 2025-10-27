import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import retrieve
import data


model_name = "Qwen/Qwen-7B"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)

documents = data.data("PeterJinGo/wiki-18-corpus", 500)

tokenized_docs = [doc.split() for doc in documents]

def answer_question(query):
    pass