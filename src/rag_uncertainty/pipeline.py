import json
import logging
from pathlib import Path
import torch

from rag_uncertainty.atomic_facts import AtomicFactGenerator

from rag_uncertainty.retrievers import build_wikipedia_retriever
from rag_uncertainty.pipeline_utils import LLM, sample_generations
from rag_uncertainty.eval import EvalEngine

logger = logging.getLogger("rag_uncertainty")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

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
    logger.debug(f"Saved {_RESULTS_JSON_PATH}")


def pipeline():
    # 1. Setup Resources
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    
    logger.info("Loading LLM")
    llm = LLM("Qwen/Qwen2.5-7B-Instruct", device)
    logger.info("LLM successfully loaded.")
    
    logger.info("Loading Wikipedia Retriever")
    retriever = build_wikipedia_retriever(
        cache_dir="bm25_index_cache",
        data_cache_dir="data",
    )
    logger.info("Wikipedia Retriever successfully loaded.")

    fact_gen = AtomicFactGenerator(llm=llm)

    # 2. Set up evaluation
    engine = EvalEngine(llm=llm, retriever=retriever, fact_gen=fact_gen)

    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    results = []

    num_categories = len(list(questions_data.keys()))

    for i, (category, questions) in enumerate(questions_data.items()):
        num_questions = len(questions)
        for j, question in enumerate(questions):
            logger.debug(f"Category ({i}/{num_categories}): {category}; Question ({j}/{num_questions}): {question}")

            logger.debug("Generating answers.")
            generations = sample_generations(
                llm=llm,
                question=question,
                retriever=retriever,
                k_ctx=5,
                n=5
            )

            logger.debug(f'Prompt used: {generations['prompt_used']}')
            for i, answers in enumerate(generations['generated_texts']):
                logger.debug(f"Answer {i}: {answers}")
            

            logger.debug("Evaluating metrics (SE, SumEigen, RAFE, and CEM)")
            metrics = engine.evaluate(generations, question)
            logger.debug("Evaluating complete.")
            
            se = metrics["semantic_entropy"]
            su = metrics["sum_eigen"]
            rafe_out = metrics["rafe"]
            logger.debug("DONE")

            # 3. Process Results
            for gen_id, answer in enumerate(generations["generated_texts"]):
                gen_info = rafe_gen_results[gen_id]
                details = gen_info["details"]
                atomic_facts = [d["claim"] for d in details]

                record = {
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "atomic_facts": atomic_facts,
                    "rafe_details": details,
                    # SE
                    "semantic_entropy_global": se["semantic_entropy"],
                    "semantic_entropy_per_gen": se["score_for_each_generation"][gen_id],
                    "semantic_entropy_truth_value": se["truth_value"],
                    # SU
                    "sum_eigen": su["U_eigv"],
                    "sum_eigen_truth_value": su["truth_value"],
                    # RAFE
                    "rafe_overall_score": rafe_out["score"],
                    "rafe_gen_score": gen_info["score"],
                    "rafe_gen_supported": gen_info["supported"],
                    "rafe_gen_not_supported": gen_info["not_supported"],
                    "rafe_gen_irrelevant": gen_info["irrelevant"],
                    "rafe_gen_total_claims": gen_info["total_claims"],
                }

                results.append(record)
            
            _write_results_json(results)

    logger.debug("Finished run.")


if __name__ == "__main__":
    pipeline()
