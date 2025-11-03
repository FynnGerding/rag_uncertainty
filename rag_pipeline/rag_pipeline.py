import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import data
import retrieve
import input_query


model_name = "gpt2"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device chosen")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("tokanizer loaded")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)
print("gpt2 loaded")
documents = data.data(1)
bm25 = retrieve.build_bm25(documents)


query = input("Question:")

while query != "quit":
    print(f"Answer: {input_query.answer_question(query, tokenizer, model, bm25, documents, 5)}")
    query = input("Question:")

print("Bye")