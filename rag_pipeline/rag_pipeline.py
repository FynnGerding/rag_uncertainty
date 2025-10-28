import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import data
import retrieve
import input_query


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

documents = data.data("PeterJinGo/wiki-18-corpus", 100)
bm25 = retrieve.build_bm25(documents)

query = input("Question:")

while query != "quit":
    print(f"Answer: {input_query.answer_question(query, tokenizer, model, bm25, documents, 5)}")
    query = input("Question:")

print("Bye")