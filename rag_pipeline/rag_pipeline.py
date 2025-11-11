import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import data
import retrieve
import input_query


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device = 'cpu'
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print('Tokenizer loaded successfully')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)

print('Model loaded successfully')

documents = data.data("wikimedia/wikipedia", 100)

print(f'Loaded {len(documents)} documents')

bm25 = retrieve.build_bm25(documents)

print('BM25 index built successfully')

print('-'*50)

query = input("Question:")

while query != "quit":
    print(f"Answer: {input_query.answer_question(query, tokenizer, model, bm25, documents, 5)}")
    query = input("Question:")

print("Bye")