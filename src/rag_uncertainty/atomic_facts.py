# Code inspired from https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py

import logging
import nltk
from typing import List, Tuple
from pydantic import BaseModel, Field

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger("rag_uncertainty")

# MARK: Schema Definition
class AtomicFactList(BaseModel):
    reasoning: str = Field(..., description="Analysis of sentence structure and distinct information units.")
    facts: List[str] = Field(..., description="List of atomic, self-contained facts.")

# MARK: Generator Class
class AtomicFactGenerator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, generation: str) -> Tuple[List[Tuple[str, List[str]]], List[int]]:
        """
        Splits text into atomic facts using structured generation.
        Returns: (List of (sentence, facts), List of paragraph break indices)
        """
        # SAFE Paper Step 1: Split response into individual facts [cite: 39]
        paragraphs = [p.strip() for p in generation.split("\n") if p.strip()]
        atomic_facts_pairs = []
        para_breaks = []
        
        current_sent_count = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(current_sent_count)
            
            # Paper uses NLTK tokenizer 
            sentences = nltk.sent_tokenize(paragraph)
            
            for sent in sentences:
                clean_sent = sent.strip()
                if not clean_sent:
                    continue

                try:
                    # Attempt structured extraction
                    extraction = self._extract_atomic_facts(clean_sent)
                    facts = extraction.facts
                    reasoning = extraction.reasoning
                except Exception as e:
                    # Fallback mechanism if the model fails to generate valid JSON
                    logger.error(f"Extraction failed for sentence: '{clean_sent[:30]}...'. Error: {e}")
                    facts = [clean_sent]
                    reasoning = "Fallback due to error."

                self._log_split(clean_sent, facts, reasoning)

                atomic_facts_pairs.append((clean_sent, facts))
                current_sent_count += 1

        return atomic_facts_pairs, para_breaks

    def _extract_atomic_facts(self, sentence: str) -> AtomicFactList:
        # Instruction aligned with SAFE decomposition logic 
        system_msg = (
            "You are an expert editor. Break the input sentence into atomic, indivisible facts.\n"
            "Rules:\n"
            "1. Split compound sentences (and, but, which) into separate claims.\n"
            "2. Keep the original wording where possible.\n"
            "3. Do not resolve pronouns (he/she/it) yet; keep them as they appear."
        )

        # Static few-shot history prevents regression/hallucination
        history = [
            {
                "role": "user", 
                "content": "Sentence: Apollo 11, commanded by Armstrong, landed safely in 1969."
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Contains three distinct info units: command, action, and date.", "facts": ["Apollo 11 was commanded by Armstrong.", "Apollo 11 landed safely.", "Apollo 11 landed in 1969."]}'
            },
            {
                "role": "user", 
                "content": "Sentence: The movie, which was released in 1999, failed at the box office."
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Relative clause provides release date; main clause describes performance.", "facts": ["The movie was released in 1999.", "The movie failed at the box office."]}'
            }
        ]

        full_prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        for msg in history:
            full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
        full_prompt += f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n<|im_start|>assistant\n"

        result, _ = self.llm.generate(
            full_prompt,
            constraint=AtomicFactList,
            temperature=0.1,
            max_new_tokens=512
        )
        
        return result

    def _log_split(self, sentence: str, facts: List[str], reasoning: str):
        log_msg = (
            f"\n{'='*60}\n"
            f"INPUT SENTENCE:\n  {sentence}\n"
            f"REASONING:\n  {reasoning}\n"
            f"ATOMIC FACTS:\n"
        )
        for i, fact in enumerate(facts, 1):
            log_msg += f"  {i}. {fact}\n"
        log_msg += f"{'='*60}"
        logger.debug(log_msg)