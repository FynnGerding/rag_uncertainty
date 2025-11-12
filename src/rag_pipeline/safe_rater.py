import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HFChatRater:
    """
    Minimal adapter that matches the methods SAFE's prompting utilities call.
    If SAFE expects different names (e.g. .complete), mirror them here.
    """
    def __init__(self, name: str, device: str = None, max_new_tokens: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**toks, do_sample=False,
                                  max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens))
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
