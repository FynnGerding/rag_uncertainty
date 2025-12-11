# Inspired by https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/

import logging
from atomic_facts import AtomicFactGenerator
from typing import Dict, Any, List

from rag_uncertainty.pipeline_utils import _qwen_prompt

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
IRRELEVANT = "IRRELEVANT"
RELEVANT = "RELEVANT"

logger = logging.getLogger("rag_uncertainty")

def _stringify(hit: Any) -> str:
    """Convert retriever results to plain strings."""
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        for k in ("page_content", "text", "content", "body", "snippet"):
            v = hit.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def revise_fact(atomic_fact: str, original_context: str, llm) -> str:
    """
    Rewrite an atomic fact so it is self-contained.
    Uses unconstrained generation but within a strict chat template.
    """
    system_msg = (
        "You are a helpful editor. Your task is to rewrite the 'Claim' to be self-contained "
        "by replacing pronouns (he, she, it, they) with specific names from the 'Context'.\n"
        "Rules:\n"
        "1. Output ONLY the rewritten sentence.\n"
        "2. Do not add new information.\n"
        "3. If the Claim cannot be resolved using the Context, return the Claim unchanged."
    )
    
    user_msg = (
        f"Context: {original_context}\n"
        f"Claim: {atomic_fact}\n"
        "Rewritten:"
    )

    prompt = _qwen_prompt(system_msg, user_msg)
    
    # We restrict the output to a single line to prevent verbose chatter
    # Regex: Any characters followed by End of String or Newline
    revised, _ = llm.generate(prompt, temperature=0.1, constraint=r"[^\n]+")
    
    revised = revised.strip()
    logger.debug(f"Fact '{atomic_fact}' was revised to: '{revised}'")
    return revised or atomic_fact.strip()


def check_relevance(question: str, atomic_fact: str, answer_context: str, llm) -> str:
    """
    Determine whether an atomic fact helps answer the user's question.
    Uses a regex constraint to force a 'Reasoning' -> 'Label' structure.
    """
    logger.debug(f"Performing relevance check of atomic fact: {atomic_fact}")
    
    system_msg = (
        "You filter atomic facts for factuality evaluation. "
        "Does the Atomic Fact help answer the Question given the context?\n"
        "If it directly addresses, supports, or contradicts the question, it is RELEVANT. "
        "If it is background, tangential, or unrelated, it is IRRELEVANT."
    )
    
    user_msg = (
        f"Question: {question}\n"
        f"Answer (for context): {answer_context}\n"
        f"Atomic fact: {atomic_fact}\n\n"
        "Respond with a short rationale (max 1 sentence) followed by the final Label."
    )

    prompt = _qwen_prompt(system_msg, user_msg)
    
    # Force the model to output: "Rationale: ... \nLabel: RELEVANT|IRRELEVANT"
    # This ensures we get Chain-of-Thought accuracy with structured parsing
    structure_regex = r"Rationale: [^\n]+\nLabel: (RELEVANT|IRRELEVANT)"
    
    result, _ = llm.generate(
        prompt, 
        temperature=0.0, 
        constraint=structure_regex,
        max_new_tokens=128
    )
    
    # Extract the label from the structured output
    if "Label: RELEVANT" in result:
        return RELEVANT
    return IRRELEVANT


def retrieve_evidence(revised_fact: str, llm, retriever, top_k: int = 5):
    """
    Generate a verification query using a constrained regex to ensure a clean single-line query.
    """
    system_msg = "You create concise Google search queries to fact-check statements."
    user_msg = (
        f"Statement: {revised_fact}\n"
        "Write a concise search query that would retrieve evidence to verify whether this statement is true.\n"
        "Query:"
    )

    prompt = _qwen_prompt(system_msg, user_msg)
    
    # Regex constraint: Non-newline characters only
    search_query, _ = llm.generate(prompt, temperature=0.1, constraint=r"[^\n]+")
    
    search_query = search_query.strip() or revised_fact

    hits = retriever.search(query=search_query, top_k=top_k)
    evidence = [_stringify(h) for h in hits][:top_k]
    return search_query, evidence


def _rate_fact(
    *,
    claim: str,
    question: str,
    answer: str,
    evidence: List[str],
    rater,
    valid_labels: List[str] | None = None,
) -> str:
    """
    Classify the claim using strict constrained generation (No parsing logic required).
    """
    logger.debug(f"Rating claim: '{claim}'")
    valid_labels = valid_labels or [SUPPORTED, NOT_SUPPORTED]
    joined_evidence = "\n".join(f"- {e}" for e in evidence if e.strip())
    
    system_msg = (
        "You are a strict factuality checker.\n"
        "Given the QUESTION, ANSWER, and EVIDENCE, classify the CLAIM as one of: "
        f"{', '.join(valid_labels)}.\n"
        "If evidence is missing or inconclusive, choose NOT_SUPPORTED."
    )
    
    user_msg = (
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n"
        f"CLAIM: {claim}\n"
        f"EVIDENCE: {joined_evidence or '(no evidence)'}\n\n"
        "Label:"
    )

    prompt = _qwen_prompt(system_msg, user_msg)
    
    # Dynamically build regex from valid labels e.g., "(SUPPORTED|NOT_SUPPORTED)"
    label_options = "|".join(valid_labels)
    constraint_regex = f"({label_options})"
    
    # The model acts as a classifier here, outputting ONLY the label
    label, _ = rater.generate(prompt, temperature=0.0, constraint=constraint_regex)
    
    return label


def rafe_factuality(
        generations: dict, 
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
            original_context = answer 
            
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