from pathlib import Path
import csv
import json

import torch

import retrievers
from pipeline_utils import load_model_and_tokenizer, sample_generations
from uncertainty_estimation import semantic_entropy, sum_eigen, safe_factuality
import data
from atomic_facts import AtomicFactGenerator

_MODULE_DIR = Path(__file__).resolve().parent
_QUESTIONS_PATH = _MODULE_DIR / "questions.json"
_RESULTS_DIR = Path.cwd() / "results"
_RESULTS_JSON_PATH = _RESULTS_DIR / "results.json"
_RESULTS_CSV_PATH = _RESULTS_DIR / "results.csv"


def _write_results_json(records):
    tmp_path = _RESULTS_JSON_PATH.with_suffix(".json.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp_path.replace(_RESULTS_JSON_PATH)
    print(f"Saved {_RESULTS_JSON_PATH} (intermediate)")

def pipeline():
    # pick device & load model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    llm = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct", device)

    # load docs (strings or dicts with "text")
    docs = data.data("wikimedia/wikipedia")

    # Option A: sparse BM25
    retriever = retrievers.BM25Retriever(docs)

    fact_gen = AtomicFactGenerator(llm=llm, is_bio=False)

    # Option B (swap to dense):
    # retriever = retrievers.ContrieverRetriever(docs, model_name="facebook/contriever-msmarco", device=device)

    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    results = []

    for category, questions in questions_data.items():
        for question in questions:
            print(f"{category}: {question}")

            print("Generation of answers...")
            generations = sample_generations(
                llm=llm,
                question=question,
                retriever=retriever,
                k_ctx=5,
                n=5,
                max_new_tokens=32,
                temperature=0.1,
                top_p=0.95,
                base_seed=0,
            )

            print("DONE")
            print("Building SE and SU...")
            se = semantic_entropy(generations, question)
            su = sum_eigen({"generated_texts": generations["generated_texts"]}, question)
            print("DONE")

            print("Building SAFE...")
            safe_out = safe_factuality(
                generations,
                question,
                llm,
                retriever=retriever,
                fact_gen=fact_gen,
                top_k=5,
                per_generation=True)

            print("DONE")

            for gen_id, answer in enumerate(generations["generated_texts"]):
                gen_info = safe_out["per_generation"][str(gen_id)]
                details = gen_info["details"]
                atomic_facts = [d["claim"] for d in details]

                record = {
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "atomic_facts": atomic_facts,
                    "safe_details": details,
                    "semantic_entropy_global": se["semantic_entropy"],
                    "semantic_entropy_per_gen": se["score_for_each_generation"][gen_id],
                    "semantic_entropy_truth_value": se["truth_value"],
                    "sum_eigen": su["U_eigv"],
                    "sum_eigen_truth_value": su["truth_value"],
                    "safe_overall_score": safe_out["overall"]["score"],
                    "safe_gen_score": gen_info["score"],
                    "safe_gen_supported": gen_info["supported"],
                    "safe_gen_not_supported": gen_info["not_supported"],
                    "safe_gen_irrelevant": gen_info["irrelevant"],
                    "safe_gen_total_claims": gen_info["total_claims"],
                }

                results.append(record)
            _write_results_json(results)

    print("\nFinished run.")

    if results:
        print(f"Final outputs saved to {_RESULTS_JSON_PATH} and {_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    pipeline()
