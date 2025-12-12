# Inspired by https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/

import logging
from typing import Dict, Any, List, Literal, Optional
import outlines
from pydantic import BaseModel, Field

from rag_uncertainty.pipeline_utils import _qwen_prompt, LLM
from rag_uncertainty.atomic_facts import AtomicFactGenerator

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
IRRELEVANT = "IRRELEVANT"
RELEVANT = "RELEVANT"

logger = logging.getLogger("rag_uncertainty")

def _get_text(hit: Any) -> str:
    """Extract text from retriever result."""
    if hasattr(hit, "text"):
        return str(hit.text)
    
    raise ValueError(f"Retrieved hit of type '{type(hit).__name__}' has no '.text' attribute. Value: {hit}")

class AtomicClaim(BaseModel):
    reasoning: str = Field(..., description="Briefly list pronouns to resolve or artifacts (like '- <fact>') to remove.")
    revised_claim: str = Field(..., description="The final, self-contained, clean sentence.")

def revise_fact(atomic_fact: str, original_context: str, llm: LLM) -> str:
    """
    Cleans and decontextualizes an atomic fact using structured generation.
    """
    system_msg = (
        "You are a precise data cleaner. Your goal is to make the 'Claim' self-contained.\n"
        "1. Replace pronouns (he/she/it/they) with specific names from 'Context'.\n"
        "2. Remove formatting artifacts (e.g., bullets, '- <fact>', 'Statement:').\n"
        "3. If the claim is already clear or cannot be resolved, return it clean but unchanged."
    )

    history = [
        # Turn 1
        {
            "role": "user",
            "content": "Context: Armstrong commanded Apollo 11.\nClaim: He landed the Eagle safely."
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "Resolve \'He\' to \'Armstrong\'.", "revised_claim": "Armstrong landed the Eagle safely."}'
        },
        # Turn 2 "Artifact"
        {
            "role": "user",
            "content": "Context: The Apollo program ended in 1972.\nClaim: - <fact> It was considered a major success."
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "Remove \'- <fact>\'. Resolve \'It\' to \'The Apollo program\'.", "revised_claim": "The Apollo program was considered a major success."}'
        },
        # Turn 3 "No Context"
        {
            "role": "user",
            "content": "Context: (No relevant context)\nClaim: The mission failed."
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "No context to resolve specific mission.", "revised_claim": "The mission failed."}'
        }
    ]

    final_user_content = f"Context: {original_context}\nClaim: {atomic_fact}"

    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    full_prompt += f"<|im_start|>user\n{final_user_content}<|im_end|>\n<|im_start|>assistant\n"

    try:
        result, _ = llm.generate(
            full_prompt, 
            constraint=AtomicClaim,
            temperature=0.1,
            max_new_tokens=256
        )
        logger.debug(f"Atomic fact: '{atomic_fact}' was revised to '{result.revised_claim}'")
        return result.revised_claim
        
    except Exception as e:
        logger.warning(f"Revision failed for fact '{atomic_fact[:30]}...': {e}")
        # Fallback
        return atomic_fact


class RelevanceDecision(BaseModel):
    rationale: str = Field(..., description="Explain why the fact helps (or doesn't help) answer the question.")
    label: Literal["RELEVANT", "IRRELEVANT"] = Field(..., description="The final verdict.")

def check_relevance(question: str, atomic_fact: str, answer_context: str, llm: LLM) -> str:
    """
    Determine whether an atomic fact helps answer the user's question.
    Uses structured generation to prevent logic errors.
    """
    system_msg = (
        "You are an impartial judge evaluating factual relevance.\n"
        "Rules:\n"
        "1. RELEVANT: The fact provides part of the answer, defines the subject, establishes necessary context (dates, locations), or directly addresses the prompt.\n"
        "2. IRRELEVANT: The fact is off-topic, completely unrelated trivia, or meta-talk (e.g., 'The text says').\n"
        "3. Contextual facts (who, what, where) ARE relevant."
    )

    history = [
        {
            "role": "user",
            "content": (
                "Question: Who is Barack Obama?\n"
                "Atomic Fact: Barack Obama served as the 44th US President.\n"
                "Task: Is this fact relevant?"
            )
        },
        {
            "role": "assistant",
            "content": '{"rationale": "This defines who the subject is, which is the core of the answer.", "label": "RELEVANT"}'
        },
        {
            "role": "user",
            "content": (
                "Question: How does photosynthesis work?\n"
                "Atomic Fact: The scientist who discovered it was born in 1850.\n"
                "Task: Is this fact relevant?"
            )
        },
        {
            "role": "assistant",
            "content": '{"rationale": "Biographical trivia about the scientist does not explain the biological process requested.", "label": "IRRELEVANT"}'
        },
        {
            "role": "user",
            "content": (
                "Question: What happened during the Apollo 13 mission?\n"
                "Atomic Fact: Apollo 13 was a lunar mission launched in 1970.\n"
                "Task: Is this fact relevant?"
            )
        },
        {
            "role": "assistant",
            "content": '{"rationale": "It establishes the subject and timeline, which is necessary context for describing the event.", "label": "RELEVANT"}'
        }
    ]
    
    final_user_content = (
        f"Question: {question}\n"
        f"Context (for reference): {answer_context}\n"
        f"Atomic Fact: {atomic_fact}\n"
        "Task: Is this fact relevant?"
    )

    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    full_prompt += f"<|im_start|>user\n{final_user_content}<|im_end|>\n<|im_start|>assistant\n"

    try:
        result, _ = llm.generate(
            full_prompt, 
            constraint=RelevanceDecision,
            temperature=0.0,
            max_new_tokens=128
        )
        
        logger.debug(f"Relevance atomic fact: '{atomic_fact}' was determined as '{result.label}'")
        return result.label

    except Exception as e:
        logger.warning(f"Relevance check failed for fact '{atomic_fact[:30]}...': {e}")
        return IRRELEVANT


class SearchQuery(BaseModel):
    query: str = Field(..., description="A neutral, keyword-optimized search string.")

def retrieve_evidence(revised_fact: str, llm: LLM, retriever, top_k: int = 5):
    """
    Generates a targeted search query and retrieves evidence.
    """
    history = [
        {"role": "user", "content": "Statement: Apollo 13 failed due to an oxygen tank explosion."},
        {"role": "assistant", "content": '{"query": "Apollo 13 oxygen tank explosion cause"}'},
        
        {"role": "user", "content": "Statement: Python was released in 1991."},
        {"role": "assistant", "content": '{"query": "Python programming language release date"}'},
    ]

    system_msg = "You generate concise Google search queries to verify statements. remove stop words."
    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    full_prompt += (
        f"<|im_start|>user\nStatement: {revised_fact}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        result, _ = llm.generate(
            full_prompt, 
            constraint=SearchQuery,
            temperature=0.1,
            max_new_tokens=64
        )
        search_query = result.query
        
    except Exception as e:
        search_query = revised_fact

    hits = retriever.search(query=search_query, top_k=top_k)
    evidence = [_get_text(h) for h in hits][:top_k]
    return search_query, evidence

class FactualityVerdict(BaseModel):
    reasoning: str = Field(
        ..., 
        description="Step-by-step comparison of the Claim vs. the Evidence. Quote the evidence if possible."
    )
    label: Literal["SUPPORTED", "NOT_SUPPORTED"] = Field(
        ..., 
        description="The final judgment. If evidence is missing/contradictory, use NOT_SUPPORTED."
    )

def _rate_fact(
    *,
    claim: str,
    question: str,
    answer: str,
    evidence: List[str],
    rater: LLM,
    valid_labels: Optional[List[str]] = None,
) -> str:
    """
    Classifies a claim using Chain-of-Thought reasoning and strict schema validation.
    """
    joined_evidence = "\n".join(f"- {e}" for e in evidence if e.strip())
    
    system_msg = (
        "You are a strict factuality judge. Your job is to verify if the 'CLAIM' is supported by the 'EVIDENCE'.\n"
        "Rules:\n"
        "1. SUPPORTED: The claim is clearly and directly backed by the provided evidence.\n"
        "2. NOT_SUPPORTED: The evidence contradicts the claim OR the evidence is missing/unrelated.\n"
        "3. Ignore your own knowledge. Rely ONLY on the provided EVIDENCE list."
    )

    # Few-Shot Example
    history = [
        {
            "role": "user",
            "content": (
                "QUESTION: Who walked on the moon?\n"
                "ANSWER: Armstrong walked on the moon.\n"
                "CLAIM: Armstrong walked on the moon.\n"
                "EVIDENCE: - Neil Armstrong was the first person to walk on the Moon in 1969.\n"
                "Task: Verify the claim."
            )
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "The evidence explicitly states Armstrong walked on the Moon. The claim matches the evidence.", "label": "SUPPORTED"}'
        },
        {
            "role": "user",
            "content": (
                "QUESTION: When was Python released?\n"
                "ANSWER: It was released in 1991.\n"
                "CLAIM: Python was released in 1991.\n"
                "EVIDENCE: - Python is a high-level programming language.\n- Guido van Rossum created it.\n"
                "Task: Verify the claim."
            )
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "The evidence confirms Python is a language and mentions the creator, but NO evidence mentions the year 1991. Therefore, it cannot be verified.", "label": "NOT_SUPPORTED"}'
        },
        {
            "role": "user",
            "content": (
                "QUESTION: Is the sky blue?\n"
                "ANSWER: The sky is blue.\n"
                "CLAIM: The sky is green.\n"
                "EVIDENCE: - Rayleigh scattering causes the sky to appear blue.\n"
                "Task: Verify the claim."
            )
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "The claim says green, but the evidence explicitly says blue. This is a direct contradiction.", "label": "NOT_SUPPORTED"}'
        }
    ]
    
    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    final_user_content = (
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n"
        f"CLAIM: {claim}\n"
        f"EVIDENCE: {joined_evidence or '(no evidence provided)'}\n"
        "Task: Verify the claim."
    )
    
    full_prompt += f"<|im_start|>user\n{final_user_content}<|im_end|>\n<|im_start|>assistant\n"

    try:
        result, _ = rater.generate(
            full_prompt, 
            constraint=FactualityVerdict,
            temperature=0.0,
            max_new_tokens=256
        )
        
        logger.debug(f"Rated claim '{claim[:30]}...' -> {result.label} (Reasoning: {result.reasoning})")
        return result.label

    except Exception as e:
        logger.warning(f"Rating failed for claim '{claim[:30]}...': {e}")
        return NOT_SUPPORTED


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
            context_text = "\n".join(_get_text(h) for h in fallback_hits).strip()

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