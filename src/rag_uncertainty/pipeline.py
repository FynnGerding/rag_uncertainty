import json
import logging
import os
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


def _dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _iter_questions(questions_data):
    for category, questions in questions_data.items():
        for question in questions:
            yield category, question


def pipeline():
    # 1. Setup Resources
    rank, world_size, local_rank = _dist_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Rank {rank}/{world_size} using device {device}")
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_rank{rank}.json"

    def _write_rank_results(records):
        tmp_path = results_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        tmp_path.replace(results_path)
        logger.debug(f"Saved {results_path}")
    
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
    engine = EvalEngine(llm=llm, retriever=retriever, fact_gen=fact_gen, device=device)

    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    results = []

    for q_idx, (category, question) in enumerate(_iter_questions(questions_data)):
        if (q_idx % world_size) != rank:
            continue

        logger.debug(f"[rank {rank}] q_idx={q_idx} category={category} question={question}")

        logger.debug("Generating answers.")
        generations = sample_generations(
            llm=llm,
            question=question,
            retriever=retriever,
            k_ctx=5,
            n=5
        )

        logger.debug(f"Prompt used: {generations.get('prompt_used', 'N/A')}")
        for k, ans in enumerate(generations['generated_texts']):
            logger.debug(f"Answer {k}: {ans}.")
        
        logger.debug("Evaluating metrics (SE, SumEigen, RAFE, and CEM)")
        metrics = engine.evaluate(generations, question)
        logger.debug("Evaluating complete.")
        
        se = metrics.get("semantic_entropy", {})
        su = metrics.get("sum_eigen", {})
        rafe_out = metrics.get("rafe", {})
        
        # Access the list of per-generation results from RAFE
        rafe_gen_results = rafe_out.get("generations", [])

        # 3. Process Results
        for gen_id, answer in enumerate(generations["generated_texts"]):
            # Retrieve the matching RAFE result for this generation
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
                "semantic_entropy_global": se.get("semantic_entropy"),
                "semantic_entropy_per_gen": se.get("score_for_each_generation", [])[gen_id] if "score_for_each_generation" in se else None,
                "semantic_entropy_truth_value": se.get("truth_value"),
                
                # SU
                "sum_eigen": su.get("U_eigv"),
                "sum_eigen_truth_value": su.get("truth_value"),
                
                # RAFE (Overall)
                "rafe_overall_score": rafe_out.get("score"),
                "rafe_overall_precision": rafe_out.get("precision"),
                
                # RAFE (Per Generation)
                "rafe_gen_score": gen_info["score"],
                "rafe_gen_supported": gen_info["supported"],
                "rafe_gen_not_supported": gen_info["not_supported"],
                "rafe_gen_irrelevant": gen_info["irrelevant"],
                "rafe_gen_total_claims": gen_info["total_claims"],
                
                # CEM Data (Our contribution)
                "cem_matrix": gen_info.get("cem_matrix"),
                "cem_evidence_scores": gen_info.get("cem_evidence_scores"),
            }

            results.append(record)
        
        _write_rank_results(results)

    logger.debug("Finished run.")


if __name__ == "__main__":
    pipeline()
