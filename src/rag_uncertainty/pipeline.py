from pathlib import Path
import json
import torch

from rag_uncertainty.atomic_facts import AtomicFactGenerator
from rag_uncertainty.data_loader import load_data
from rag_uncertainty.retrievers import build_wikipedia_retriever
from rag_uncertainty.pipeline_utils import load_model_and_tokenizer, sample_generations
from rag_uncertainty.eval import EvalEngine

_MODULE_DIR = Path(__file__).resolve().parent
_QUESTIONS_PATH = _MODULE_DIR / "questions.json"
_RESULTS_DIR = Path.cwd() / "results"
_RESULTS_JSON_PATH = _RESULTS_DIR / "results.json"


def _write_results_json(records):
    tmp_path = _RESULTS_JSON_PATH.with_suffix(".json.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp_path.replace(_RESULTS_JSON_PATH)
    print(f"Saved {_RESULTS_JSON_PATH}")


def pipeline():
    # 1. Setup Resources
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    
    llm = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct", device)
    
    retriever = build_wikipedia_retriever(
        data_loader=load_data,
        cache_dir="bm25_index_cache",
        data_cache_dir="data",
    )

    fact_gen = AtomicFactGenerator(llm=llm, is_bio=False)

    engine = EvalEngine(llm=llm, retriever=retriever, fact_gen=fact_gen)

    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    results = []

    for category, questions in questions_data.items():
        for question in questions:
            print(f"\n{category}: {question}")

            print("Generating answers...")
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

            print("Evaluating metrics (SE, SumEigen, SAFE)...")
            metrics = engine.evaluate(generations, question)
            
            se = metrics["semantic_entropy"]
            su = metrics["sum_eigen"]
            safe_out = metrics["safe"]
            print("DONE")

            # 3. Process Results
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
                    # SE
                    "semantic_entropy_global": se["semantic_entropy"],
                    "semantic_entropy_per_gen": se["score_for_each_generation"][gen_id],
                    "semantic_entropy_truth_value": se["truth_value"],
                    # SU
                    "sum_eigen": su["U_eigv"],
                    "sum_eigen_truth_value": su["truth_value"],
                    # SAFE
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


if __name__ == "__main__":
    pipeline()