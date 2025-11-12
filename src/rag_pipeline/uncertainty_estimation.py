from TruthTorchLM.truth_methods.semantic_entropy import SemanticEntropy
from TruthTorchLM.truth_methods.sum_eigen_uncertainty import SumEigenUncertainty

import third_party
from rag_pipeline.atomic_facts import AtomicFactGenerator
from typing import Dict, Any, List, Tuple

def semantic_entropy(generations, question, **kwargs):
    """
    Compute Semantic Entropy from externally provided generations.

    Parameters
    ----------
    generations : dict
        Must include:
          - "generated_texts": List[str]
          - "logprobs": List[List[float]]  # token-level logprobs per generation
        (Any extra fields are ignored.)
    question : str
        The original question/prompt (used for entailment clustering).
    **kwargs :
        Passed directly to SemanticEntropy(...), e.g.:
          scoring_function, number_of_generations,
          model_for_entailment, tokenizer_for_entailment,
          entailment_model_device, batch_generation.

    Returns
    -------
    dict
        Matches SemanticEntropy.forward_api(...) output:
          {
            "truth_value": float,
            "semantic_entropy": float,
            "score_for_each_generation": List[float],
            "generated_texts": List[str],
            "clusters": List[Set[str]],
          }
    """
    if "generated_texts" not in generations:
        raise ValueError('`generations` must contain key "generated_texts".')
    if "logprobs" not in generations:
        raise ValueError('`generations` must contain key "logprobs" for semantic entropy.')

    kwargs.setdefault("entailment_model_device", "cpu")
    se = SemanticEntropy(**kwargs)
    # Because sampling is external, we only pass the precomputed dict.
    return se.forward_api(
        model="",
        messages=[],
        generated_text="",
        question=question,
        sampled_generations_dict=generations,
    )


def sum_eigen(generations, question, **kwargs):
    """
    Compute Sum Eigen Uncertainty (U_eigv) from externally provided generations.

    Parameters
    ----------
    generations : dict
        Must include:
          - "generated_texts": List[str]
        (Any extra fields are ignored.)
    question : str
        The original question/prompt (used for similarity).
    **kwargs :
        Passed directly to SumEigenUncertainty(...), e.g.:
          method_for_similarity ("semantic" | "jaccard"),
          number_of_generations, temperature,
          model_for_entailment, tokenizer_for_entailment,
          entailment_model_device, batch_generation.

    Returns
    -------
    dict
        Matches SumEigenUncertainty.forward_api(...) output:
          {
            "U_eigv": float,
            "generated_texts": List[str],
            "truth_value": float,
          }
    """
    if "generated_texts" not in generations:
        raise ValueError('`generations` must contain key "generated_texts".')

    kwargs.setdefault("entailment_model_device", "cpu")
    seu = SumEigenUncertainty(**kwargs)
    # Because sampling is external, we only pass the precomputed dict.
    return seu.forward_api(
        model="",
        messages=[],
        generated_text="",
        question=question,
        sampled_generations_dict=generations,
    )

# Adaptation from https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/search_augmented_factuality_eval.py
SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
IRRELEVANT = "IRRELEVANT"

def _get_text(gen_item: Any) -> str:
    """Accepts plain string or dict with text/answer fields."""
    if isinstance(gen_item, str):
        return gen_item
    if isinstance(gen_item, dict):
        for k in ("text", "answer", "generation"):
            if k in gen_item and isinstance(gen_item[k], str):
                return gen_item[k]
    raise TypeError(f"Invalid generation item: {type(gen_item)}")


def _stringify(hit: Any) -> str:
    """Convert retriever results to plain strings."""
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        for key in ("text", "content", "body", "snippet"):
            if key in hit and isinstance(hit[key], str):
                return hit[key]
    return str(hit)


def _pick_label(raw: str) -> str:
    raw = raw.strip().upper()
    if "SUPPORTED" in raw and "NOT" not in raw:
        return SUPPORTED
    if "NOT" in raw:
        return NOT_SUPPORTED
    if "IRRELEV" in raw:
        return IRRELEVANT
    return IRRELEVANT


def _score(supported: int, not_supported: int) -> float:
    denom = supported + not_supported
    return 1.0 if denom == 0 else supported / denom


def _rate_fact(
    *,
    claim: str,
    question: str,
    answer: str,
    evidence: List[str],
    rater,
) -> str:
    """Ask rater (HFChatRater or compatible) to check if claim is supported."""
    joined_evidence = "\n".join(f"- {e}" for e in evidence if e.strip())
    prompt = f"""You are a factuality checker.
Given the QUESTION, ANSWER, and EVIDENCE, classify the CLAIM as one of:
SUPPORTED, NOT_SUPPORTED, or IRRELEVANT.

QUESTION:
{question}

ANSWER:
{answer}

CLAIM:
{claim}

EVIDENCE:
{joined_evidence or '(no evidence)'}

Label:"""
    result = rater.generate(prompt, temperature=0)
    return _pick_label(result)


def safe_factuality(
        generations: dict, # {"generated_texts": List[str], "logprobs": ...}
        question: str,
        llm,
        *,
        rater,
        retriever,
        top_k: int = 5,
        per_generation: bool = True,
    ) -> Dict[str, Any]:
    fact_gen = AtomicFactGenerator(llm=llm)

    overall_sup = overall_not = overall_irrel = 0
    per_gen: Dict[str, Any] = {}

    for gid, answer in enumerate(generations["generated_texts"]):
        # extract atomic claims
        atomic_pairs, _ = fact_gen.run(answer)
        claims = [c for _, facts in atomic_pairs for c in facts if c and c.strip()]
        # optional: de-dup while preserving order
        claims = list(dict.fromkeys(claims))

        g_sup = g_not = g_irrel = 0
        details = []

        for claim in claims:
            hits = retriever.search(query=claim, top_k=top_k)
            evidence = [_stringify(h) for h in hits][:top_k]

            label = _rate_fact(
                claim=claim,
                question=question,
                answer=answer,
                evidence=evidence,
                rater=rater,
            )

            if label == SUPPORTED:
                g_sup += 1
            elif label == NOT_SUPPORTED:
                g_not += 1
            else:
                g_irrel += 1

            details.append({"claim": claim, "label": label, "evidence": evidence})

        if per_generation:
            total = g_sup + g_not + g_irrel
            per_gen[str(gid)] = {
                "score": _score(g_sup, g_not),
                "supported": g_sup,
                "not_supported": g_not,
                "irrelevant": g_irrel,
                "total_claims": total,
                "details": details,
            }

        overall_sup += g_sup
        overall_not += g_not
        overall_irrel += g_irrel

    overall_total = overall_sup + overall_not + overall_irrel
    overall = {
        "score": _score(overall_sup, overall_not),
        "supported": overall_sup,
        "not_supported": overall_not,
        "irrelevant": overall_irrel,
        "total_claims": overall_total,
    }

    return {"overall": overall, **({"per_generation": per_gen} if per_generation else {})}