import torch

import retrievers
from pipeline_utils import load_model_and_tokenizer, sample_generations
from uncertainty_estimation import semantic_entropy, sum_eigen, safe_factuality
import data
from atomic_facts import AtomicFactGenerator
import csv
import json


def pipeline():
    # pick device & load model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    llm = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device)

    # load docs (strings or dicts with "text")
    docs = data.data("wikimedia/wikipedia", 100)

    # Option A: sparse BM25
    retriever = retrievers.BM25Retriever(docs)

    fact_gen = AtomicFactGenerator(llm=llm, is_bio=False)

    # Option B (swap to dense):
    # retriever = retrievers.ContrieverRetriever(docs, model_name="facebook/contriever-msmarco", device=device)

    with open("questions.json", "r", encoding="utf-8") as f:
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

    with open(results.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results.json")

    if results:
        fieldnames = [
            "category",
            "question",
            "answer",
            "atomic_facts",
            "semantic_entropy_global",
            "semantic_entropy_per_gen",
            "semantic_entropy_truth_value",
            "sum_eigen",
            "sum_eigen_truth_value",
            "safe_overall_score",
            "safe_gen_score",
            "safe_gen_supported",
            "safe_gen_not_supported",
            "safe_gen_irrelevant",
            "safe_gen_total_claims",
        ]

        with open("results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rec in results:
                row = {
                    "category": rec["category"],
                    "question": rec["question"],
                    "answer": rec["answer"],
                    "atomic_facts": json.dumps(rec["atomic_facts"], ensure_ascii=False),
                    "semantic_entropy_global": rec["semantic_entropy_global"],
                    "semantic_entropy_per_gen": rec["semantic_entropy_per_gen"],
                    "semantic_entropy_truth_value": rec["semantic_entropy_truth_value"],
                    "sum_eigen": rec["sum_eigen"],
                    "sum_eigen_truth_value": rec["sum_eigen_truth_value"],
                    "safe_overall_score": rec["safe_overall_score"],
                    "safe_gen_score": rec["safe_gen_score"],
                    "safe_gen_supported": rec["safe_gen_supported"],
                    "safe_gen_not_supported": rec["safe_gen_not_supported"],
                    "safe_gen_irrelevant": rec["safe_gen_irrelevant"],
                    "safe_gen_total_claims": rec["safe_gen_total_claims"],
                }
                writer.writerow(row)

        print(f"Saved results.csv")


if __name__ == "__main__":
    pipeline()
