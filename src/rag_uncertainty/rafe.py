# Inspired by https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/

import logging
from typing import Dict, Any, List, Literal, Optional
import outlines
from pydantic import BaseModel, Field

from rag_uncertainty.pipeline_utils import LLM
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

# MARK: Fact Revision (Self-Containment)

class AtomicClaim(BaseModel):
    reasoning: str = Field(..., description="Plan: 1. Identify pronouns/ambiguities. 2. Find their referents in Context.")
    revised_claim: str = Field(..., description="The final, self-contained sentence with full entity names.")

def revise_fact(atomic_fact: str, original_context: str, llm: LLM) -> str:
    """
    Cleans an atomic fact by resolving pronouns and making it self-contained.
    """
    system_msg = (
        "You are an expert editor. Your goal is to make the 'Claim' self-contained and searchable.\n"
        "1. Aggressively resolve pronouns (he/she/it/they) to full names from 'Context'.\n"
        "2. Replace vague terms like 'the mission' or 'the company' with specific entities (e.g., 'Apollo 13').\n"
        "3. Remove structural artifacts (e.g., '- <fact>', 'However,').\n"
        "4. If the claim is already clear, return it unchanged."
    )

    history = [
        {
            "role": "user",
            "content": "Context: Mattingly was removed from the Apollo 13 crew.\nClaim: He was replaced by Swigert."
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "Resolve \'He\' to \'Mattingly\'.", "revised_claim": "Mattingly was replaced by Swigert."}'
        },
        {
            "role": "user",
            "content": "Context: The Apollo 13 mission failed.\nClaim: However, the mission is considered a 'successful failure'."
        },
        {
            "role": "assistant",
            "content": '{"reasoning": "Remove \'However,\'. Resolve \'the mission\' to \'Apollo 13\'.", "revised_claim": "Apollo 13 is considered a \'successful failure\'."}'
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
        logger.debug(f"Revised: '{atomic_fact}' -> '{result.revised_claim}'")
        return result.revised_claim
    except Exception as e:
        logger.warning(f"Revision failed: {e}")
        return atomic_fact


# MARK: Relevance Check (Verifiability Filter)

class RelevanceDecision(BaseModel):
    rationale: str = Field(..., description="Does this claim describe the world/subject? Or is it meta-talk?")
    label: Literal["RELEVANT", "IRRELEVANT"] = Field(..., description="The final verdict.")

def check_relevance(question: str, atomic_fact: str, answer_context: str, llm: LLM) -> str:
    """
    Determine if a fact is a verifiable claim about the world (RELEVANT) or meta-talk (IRRELEVANT).
    """
    system_msg = (
        "You are a filter for an automated fact-checking system. "
        "Your goal is to label the 'Atomic Fact' based on whether it is a verifiable claim about the world.\n"
        "Labels:\n"
        "1. RELEVANT: Any claim about the subject matter (definitions, dates, events, background info). "
        "Even if it is basic or generic, if it is about the topic, it is RELEVANT.\n"
        "2. IRRELEVANT: \n"
        "   - Meta-statements about the text (e.g., 'The text says', 'details are not provided').\n"
        "   - Refusals (e.g., 'I cannot answer').\n"
        "   - Subjective transitions (e.g., 'Let's look at')."
    )

    history = [
        # Case 1: Background info -> RELEVANT
        {
            "role": "user",
            "content": "Question: What happened in Apollo 13?\nAtomic Fact: Apollo 13 was a lunar mission.\nTask: Label relevance."
        },
        {
            "role": "assistant",
            "content": '{"rationale": "It is a factual claim describing the Apollo 13 mission (the subject).", "label": "RELEVANT"}'
        },
        # Case 2: Meta-statement / Missing info -> IRRELEVANT
        {
            "role": "user",
            "content": "Question: What happened in Apollo 13?\nAtomic Fact: The specific details of the mission are not provided.\nTask: Label relevance."
        },
        {
            "role": "assistant",
            "content": '{"rationale": "This describes the lack of information in the text, not the mission itself. It is a meta-statement.", "label": "IRRELEVANT"}'
        }
    ]
    
    final_user_content = (
        f"Question: {question}\n"
        f"Atomic Fact: {atomic_fact}\n"
        "Task: Label relevance."
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
        logger.debug(f"Relevance: '{atomic_fact}' -> '{result.label}'")
        return result.label
    except Exception as e:
        logger.warning(f"Relevance check failed: {e}")
        return IRRELEVANT

# MARK: Evidence Retrieval (Query Generation)

class SearchQuery(BaseModel):
    query: str = Field(..., description="A neutral, keyword-optimized search string.")

def retrieve_evidence(revised_fact: str, llm: LLM, retriever, top_k: int = 5):
    """
    Generates a targeted search query from the fact, then retrieves evidence.
    """
    history = [
        {"role": "user", "content": "Statement: Apollo 13 failed due to an oxygen tank explosion."},
        {"role": "assistant", "content": '{"query": "Apollo 13 oxygen tank explosion cause"}'},
        {"role": "user", "content": "Statement: Python was released in 1991."},
        {"role": "assistant", "content": '{"query": "Python programming language release date"}'},
    ]

    system_msg = "You generate concise Google search queries to verify statements. Remove stop words."
    full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    for msg in history:
        full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    full_prompt += f"<|im_start|>user\nStatement: {revised_fact}<|im_end|>\n<|im_start|>assistant\n"

    try:
        result, _ = llm.generate(
            full_prompt, 
            constraint=SearchQuery,
            temperature=0.1,
            max_new_tokens=64
        )
        search_query = result.query
    except Exception:
        search_query = revised_fact

    hits = retriever.search(query=search_query, top_k=top_k)
    evidence = [_get_text(h) for h in hits][:top_k]
    return search_query, evidence

# MARK: Fact Rating

class FactualityVerdict(BaseModel):
    reasoning: str = Field(..., description="Step-by-step comparison of the Claim vs. the Evidence.")
    label: Literal["SUPPORTED", "NOT_SUPPORTED"] = Field(..., description="Final verdict.")

def _rate_fact(
    *,
    claim: str,
    question: str,
    answer: str,
    evidence: List[str],
    rater: LLM
    ) -> str:
    """
    Classifies a claim as SUPPORTED or NOT_SUPPORTED based on evidence.
    """
    joined_evidence = "\n".join(f"- {e}" for e in evidence if e.strip())
    
    system_msg = (
        "You are a strict factuality judge. Your job is to verify if the 'CLAIM' is supported by the 'EVIDENCE'.\n"
        "Rules:\n"
        "1. SUPPORTED: The claim is clearly and directly backed by the provided evidence.\n"
        "2. NOT_SUPPORTED: The evidence contradicts the claim OR the evidence is missing/unrelated.\n"
        "3. Ignore your own knowledge. Rely ONLY on the provided EVIDENCE list."
    )

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
            "content": '{"reasoning": "The evidence confirms Python is a language, but NO evidence mentions the year 1991. Missing evidence = Not Supported.", "label": "NOT_SUPPORTED"}'
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
        logger.debug(f"Rating: '{claim}' -> {result.label}\n")
        return result.label
    except Exception as e:
        logger.warning(f"Rating failed: {e}")
        return NOT_SUPPORTED

# MARK: RAFE loop

def rafe_factuality(
        generations: dict, 
        question: str,
        llm,
        *,
        retriever,
        fact_gen: AtomicFactGenerator,
        top_k: int = 5,
        K: int = 10,  # F1@K parameter (human-preferred fact count)
    ) -> Dict[str, Any]:

    f1_scores = []
    total_sup = 0
    total_not = 0
    total_irrel = 0

    for answer in generations["generated_texts"]:
        atomic_pairs, _ = fact_gen.run(answer)
        
        # Flatten and deduplicate atomic facts
        unique_facts = {f.strip() for _, facts in atomic_pairs for f in facts if f.strip()}
        
        g_sup = 0
        g_not = 0

        for raw_claim in unique_facts:
            # 1. Revise (Self-Containment)
            revised_claim = revise_fact(raw_claim, answer, llm)
            
            # 2. Check Relevance (Verifiability)
            if check_relevance(question, revised_claim, answer, llm) == IRRELEVANT:
                total_irrel += 1
                continue

            # 3. Retrieve Evidence (Query Generation)
            _, evidence = retrieve_evidence(
                revised_claim, llm=llm, retriever=retriever, top_k=top_k
            )

            # 4. Rate Fact
            label = _rate_fact(
                claim=revised_claim,
                question=question,
                answer=answer,
                evidence=evidence,
                rater=llm
            )

            if label == SUPPORTED:
                g_sup += 1
            else:
                g_not += 1

        # Calculate F1@K for this generation
        # Precision: Proportion of verified facts among all relevant facts generated
        precision = g_sup / (g_sup + g_not + 1e-9)
        
        # Recall: Proportion of supported facts against ideal length K
        recall = min(g_sup / K, 1.0)
        
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        f1_scores.append(f1)
        total_sup += g_sup
        total_not += g_not

    # Aggregate Metrics
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    overall_precision = total_sup / (total_sup + total_not + 1e-9)

    return {
        "score": avg_f1,
        "supported": total_sup,
        "not_supported": total_not,
        "irrelevant": total_irrel,
        "precision": overall_precision,
    }
