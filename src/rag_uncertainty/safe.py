# Inspired by https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/search_augmented_factuality_eval.py

from atomic_facts import AtomicFactGenerator
from typing import Dict, Any, List

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
IRRELEVANT = "IRRELEVANT"
RELEVANT = "RELEVANT"

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


def _pick_relevance(raw: str) -> str:
    raw = raw.strip().upper()
    if "IRRELEV" in raw or "NOT RELEVANT" in raw or "OFF-TOPIC" in raw:
        print(f"Decision: '{raw}' got assigned to IRRELEVANT")
        return IRRELEVANT
    print(f"Decision: '{raw}' got assigned to RELEVANT")
    return RELEVANT


def revise_fact(atomic_fact: str, original_context: str, llm) -> str:
    """
    Rewrite an atomic fact so it is self-contained, replacing vague references with concrete entities.
    Falls back to the original fact if the model does not return a revision.
    """
    prompt = f"""Task: Rewrite the "Claim" to be self-contained by replacing pronouns (he, she, it, they) with specific names from the "Context".
Rules:
1. Output ONLY the rewritten sentence.
2. Do not add new information.
3. If the Claim cannot be resolved using the Context, return the Claim unchanged.

Context: The Apollo 13 mission was a critical lunar mission. It failed to land.
Claim: It failed to land.
Rewritten: The Apollo 13 mission failed to land.

Context: Steve Jobs co-founded Apple.
Claim: He was forced out in 1985.
Rewritten: Steve Jobs was forced out in 1985.

Context: {original_context}
Claim: {atomic_fact}
Rewritten:"""
    revised, _ = llm.generate(prompt, temperature=0.1)
    revised = revised.strip()
    print(f"Fact '{atomic_fact}' was revised to: '{revised}'")
    return revised or atomic_fact.strip()


def check_relevance(question: str, atomic_fact: str, answer_context: str, llm) -> str:
    """
    Determine whether an atomic fact helps answer the user's question.
    Returns RELEVANT or IRRELEVANT.
    """
    print(f'Performing relevance check of atomic fact: {atomic_fact}')
    prompt = f"""You filter atomic facts for factuality evaluation.
Question: {question}
Answer (for context): {answer_context}
Atomic fact: {atomic_fact}

Does this fact help answer the question? If it directly addresses, supports, or contradicts the question, it is RELEVANT. If it is background, tangential, or unrelated, it is IRRELEVANT.
Respond with a short rationale followed by 'Label: RELEVANT' or 'Label: IRRELEVANT'."""
    result, _ = llm.generate(prompt, temperature=0.0001, max_new_tokens=64) # Temperature to (near) zero as in SAFE paper
    return _pick_relevance(result)


def retrieve_evidence(revised_fact: str, llm, retriever, top_k: int = 5):
    """
    Generate a verification query for a fact and retrieve evidence snippets.
    """
    query_prompt = (
        "You create search queries to fact-check statements.\n"
        f"Statement: {revised_fact}\n"
        "Write a concise search query that would retrieve evidence to verify whether this statement is true.\n"
        "Query:"
    )
    search_query, _ = llm.generate(query_prompt, temperature=0.1)
    search_query = search_query.strip() or revised_fact

    hits = retriever.search(query=search_query, top_k=top_k)
    evidence = [_stringify(h) for h in hits][:top_k]
    return search_query, evidence


def calculate_f1_at_k(supported_count: int, not_supported_count: int, K: int = 20) -> float:
    """
    Compute F1@K as described in SAFE: precision ignores irrelevant facts, recall is capped by K.
    """
    precision = supported_count / (supported_count + not_supported_count + 1e-9)
    recall = min(supported_count / K, 1.0)
    denom = precision + recall
    return 0.0 if denom == 0 else 2 * (precision * recall) / denom


def _rate_fact(
    *,
    claim: str,
    question: str,
    answer: str,
    evidence: List[str],
    rater,
    valid_labels: List[str] | None = None,
) -> str:
    print(f"Rating claim: '{claim}'")
    """Ask rater to check if claim is supported."""
    valid_labels = valid_labels or [SUPPORTED, NOT_SUPPORTED]
    joined_evidence = "\n".join(f"- {e}" for e in evidence if e.strip())
    prompt = f"""You are a factuality checker.
Given the QUESTION, ANSWER, and EVIDENCE, classify the CLAIM as one of:
SUPPORTED or NOT_SUPPORTED. If evidence is missing or inconclusive, choose NOT_SUPPORTED.

QUESTION:
{question}

ANSWER:
{answer}

CLAIM:
{claim}

EVIDENCE:
{joined_evidence or '(no evidence)'}

Label (SUPPORTED or NOT_SUPPORTED only):"""
    result, _ = rater.generate(prompt, temperature=0.01)
    picked = _pick_label(result)
    if picked not in valid_labels:
        return NOT_SUPPORTED
    return picked


def safe_factuality(
        generations: dict, # {"generated_texts": List[str], "logprobs": ...}
        question: str,
        llm,
        *,
        retriever,
        fact_gen : AtomicFactGenerator,
        top_k: int = 5,
        per_generation: bool = True,
    ) -> Dict[str, Any]:

    overall_sup = overall_not = overall_irrel = 0
    overall_context_relevant = 0
    per_gen: Dict[str, Any] = {}

    for gid, answer in enumerate(generations["generated_texts"]):
        atomic_pairs, _ = fact_gen.run(answer)

        fact_entries = []
        for sent, facts in atomic_pairs:
            for fact in facts:
                fact_text = fact.strip()
                if fact_text:
                    fact_entries.append({"fact": fact_text, "context": sent})

        deduped_facts = []
        seen = set()
        for entry in fact_entries:
            if entry["fact"] in seen:
                continue
            seen.add(entry["fact"])
            deduped_facts.append(entry)

        g_sup = g_not = g_irrel = 0
        details = []
        collected_evidence: List[str] = []

        for entry in deduped_facts:
            raw_claim = entry["fact"]
            original_context = answer  # use full answer for pronoun resolution
            revised_claim = revise_fact(raw_claim, original_context, llm)
            relevance = check_relevance(
                question=question,
                atomic_fact=revised_claim,
                answer_context=answer,
                llm=llm,
            )

            if relevance == IRRELEVANT:
                g_irrel += 1
                details.append(
                    {
                        "claim": revised_claim,
                        "original_claim": raw_claim,
                        "relevance": relevance,
                        "label": IRRELEVANT,
                        "search_query": None,
                        "evidence": [],
                    }
                )
                continue

            search_query, evidence = retrieve_evidence(
                revised_claim, llm=llm, retriever=retriever, top_k=top_k
            )
            collected_evidence.extend(evidence)

            label = _rate_fact(
                claim=revised_claim,
                question=question,
                answer=answer,
                evidence=evidence,
                rater=llm,
                valid_labels=[SUPPORTED, NOT_SUPPORTED],
            )

            if label == SUPPORTED:
                g_sup += 1
            else:
                g_not += 1

            details.append(
                {
                    "claim": revised_claim,
                    "original_claim": raw_claim,
                    "relevance": relevance,
                    "label": label,
                    "search_query": search_query,
                    "evidence": evidence,
                }
            )

        # Build recall denominator from retrieved context
        context_text = "\n".join(collected_evidence).strip()
        if not context_text:
            # fallback: retrieve using the question to build context
            fallback_hits = retriever.search(query=question, top_k=top_k)
            context_text = "\n".join(_stringify(h) for h in fallback_hits).strip()

        total_relevant_context_facts = 0
        if context_text:
            context_atomic_pairs, _ = fact_gen.run(context_text)
            context_claims = [c for _, facts in context_atomic_pairs for c in facts if c]
            for c_claim in context_claims:
                rel = check_relevance(question, c_claim, context_text, llm)
                if rel == RELEVANT:
                    total_relevant_context_facts += 1
        recall_denom = max(total_relevant_context_facts, 1)

        precision = g_sup / (g_sup + g_not + 1e-9)
        recall = g_sup / recall_denom
        f1_score = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

        if per_generation:
            total = g_sup + g_not + g_irrel
            per_gen[str(gid)] = {
                "score": f1_score,
                "supported": g_sup,
                "not_supported": g_not,
                "irrelevant": g_irrel,
                "context_relevant_facts": total_relevant_context_facts,
                "precision": precision,
                "recall": recall,
                "total_claims": total,
                "details": details,
            }

        overall_sup += g_sup
        overall_not += g_not
        overall_irrel += g_irrel
        overall_context_relevant += total_relevant_context_facts

    overall_total = overall_sup + overall_not + overall_irrel
    overall_precision = overall_sup / (overall_sup + overall_not + 1e-9)
    overall_recall = overall_sup / max(overall_context_relevant, 1)
    overall_f1 = 0.0 if (overall_precision + overall_recall) == 0 else 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    overall = {
        "score": overall_f1,
        "supported": overall_sup,
        "not_supported": overall_not,
        "irrelevant": overall_irrel,
        "context_relevant_facts": overall_context_relevant,
        "precision": overall_precision,
        "recall": overall_recall,
        "total_claims": overall_total,
    }

    return {"overall": overall, **({"per_generation": per_gen} if per_generation else {})}