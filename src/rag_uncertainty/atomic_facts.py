# Code taken from https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py
# Simplified: assumes llm is always provided (no fallback logic)
import json
import re
import string
from pathlib import Path
import spacy
import nltk
from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
from rank_bm25 import BM25Okapi

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_DEMOS_PATH = Path(__file__).resolve().with_name("demos.json")
with open(_DEMOS_PATH, "r", encoding="utf-8") as file:
    _DEMOS = json.load(file)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


_MONTHS = [m.lower() for m in [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]]


def _is_int(tok: str) -> bool:
    try:
        int(tok)
        return True
    except Exception:
        return False


def is_date(text: str) -> bool:
    text = normalize_answer(text)
    toks = text.split()
    if not toks:
        return False
    for token in toks:
        if (not _is_int(token)) and token not in _MONTHS:
            return False
    return True


_NUM_RE = re.compile(r"\b\d+\b")


def extract_numeric_values(text: str):
    return set(_NUM_RE.findall(text or ""))


def detect_initials(text: str):
    return re.findall(r"[A-Z]\.\s?[A-Z]\.", text or "")


def _safe_any(iterable):
    try:
        return any(iterable)
    except Exception:
        return False


def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not _safe_any(initial in s for s in curr_sentences):
            parts = [t.strip() for t in initial.split(".") if t.strip()]
            if len(parts) == 2:
                alpha1, alpha2 = parts
                for i, (s1, s2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                    if s1.endswith(alpha1 + ".") and s2.startswith(alpha2 + "."):
                        curr_sentences = curr_sentences[:i] + [s1 + " " + s2] + curr_sentences[i + 2:]
                        break

    sentences = []
    combine_with_previous = False
    for idx, sent in enumerate(curr_sentences):
        words = sent.split()
        if len(words) <= 1 and idx == 0:
            combine_with_previous = True
            sentences.append(sent)
        elif len(words) <= 1:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif sent and sent[0].isalpha() and not sent[0].isupper() and idx > 0:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            sentences.append(sent)
    return sentences


def sent_tokenize(text: str):
    return _nltk_sent_tokenize(text)


def text_to_sentences(text: str):
    if not text:
        return []
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    facts = []
    for line in lines:
        line = re.sub(r"^(\-|\*|\d+\.)\s*", "", line).strip()
        if not line:
            continue
        if not line.endswith("."):
            line += "."
        facts.append(line)
    if len(facts) <= 1 and ("\n" not in text.strip()):
        chunks = re.split(r";|\sand\s(?![^()]*\))", text.strip())
        if len(chunks) > 1:
            facts = [c.strip().rstrip(".") + "." for c in chunks if c.strip()]
    return facts


def best_demos(query: str, bm25, demos_sents: list[str], k: int):
    tokenized_query = query.split(" ")
    return bm25.get_top_n(tokenized_query, demos_sents, k)


def _load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("!!! spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")


def detect_entities(text: str, nlp):
    entities = set()

    def _add(text_):
        if "-" in text_:
            for t in text_.split("-"):
                entities.add(t.strip())
        else:
            entities.add(text_)

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            if is_date(ent.text):
                _add(ent.text)
            else:
                for tok in ent.text.split():
                    if is_date(tok):
                        _add(tok)

    for new_ent in extract_numeric_values(text):
        if not any(new_ent in ent for ent in entities):
            entities.add(new_ent)
    return entities


def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", "appointed.", "characterized.", "described.", "known.", "member.", "advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, list(facts)])

    new_atomic_facts = []
    for sent, facts in atomic_facts:
        entities = detect_entities(sent, nlp)
        covered = set()
        cleaned = []
        for i, fact in enumerate(facts):
            if any(fact.endswith(v) for v in verbs) and not any(fact.endswith(v) for v in permitted_verbs):
                if any(fact[:-1] in other for j, other in enumerate(facts) if j != i):
                    continue
            f_ents = detect_entities(fact, nlp)
            covered |= set(e for e in f_ents if e in entities)
            new_ents = f_ents - entities
            if new_ents:
                ok = True
                for ne in new_ents:
                    pre = None
                    for e in entities:
                        if e.startswith(ne):
                            pre = e
                            break
                    if pre is None:
                        ok = False
                        break
                    fact = fact.replace(ne, pre)
                    covered.add(pre)
                if not ok:
                    continue
            if fact not in cleaned:
                cleaned.append(fact)

        if entities and covered != entities:
            cleaned = facts
        new_atomic_facts.append((sent, cleaned))
    return new_atomic_facts, new_para_breaks


class AtomicFactGenerator:
    """
    Simplified class that breaks long-form text into atomic claims
    """
    def __init__(self, llm, demos=None, is_bio=True):
        assert llm is not None, "An LLM must be provided"
        self.is_bio = bool(is_bio)
        self.llm = llm
        self.demos = demos or dict(_DEMOS)
        self.demo_keys = list(self.demos.keys())
        self.nlp = _load_spacy()
        tokenized_corpus = [doc.split(" ") for doc in self.demo_keys]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def run(self, generation: str):
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [p.strip() for p in generation.split("\n") if p.strip()]
        return self._get_atomic_facts_from_paragraph(paragraphs)

    def _get_atomic_facts_from_paragraph(self, paragraphs):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))
            initials = detect_initials(paragraph)
            sents = fix_sentence_splitter(sent_tokenize(paragraph), initials)
            sentences += sents

        atoms = self._get_init_atomic_facts_from_sentences(sentences)

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            atomic_facts_pairs.append((sent, atoms.get(sent, [])))

        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(
                atomic_facts_pairs, list(para_breaks), self.nlp
            )

        return atomic_facts_pairs, para_breaks

    def _get_init_atomic_facts_from_sentences(self, sentences):
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            k = 1 if self.is_bio else 0
            n = 7 if self.is_bio else 8
            prompt_parts = [
                "Break the sentence into independent, minimal facts.",
                "Return one fact per line as '- <fact>.'",
                "Avoid combining multiple claims in one line.",
                ""
            ]

            head_keys = self.demo_keys[:n]
            for ex in head_keys:
                prompt_parts.append(f"Sentence: {ex}")
                for fact in self.demos[ex]:
                    prompt_parts.append(f"- {fact}")
                prompt_parts.append("")

            if k > 0 and len(self.demo_keys) > 0:
                top_matches = best_demos(sentence, self.bm25, self.demo_keys, k)
                for ex in top_matches:
                    prompt_parts.append(f"Sentence: {ex}")
                    for fact in self.demos[ex]:
                        prompt_parts.append(f"- {fact}")
                    prompt_parts.append("")

            prompt_parts.append(f"Sentence: {sentence}")
            prompt = "\n".join(prompt_parts).strip() + "\n- "

            #print(f'Prompt:\n\n{prompt}')
            output, _ = self.llm.generate(prompt)
            #print(f'\nOutput:\n{output}')

            facts = text_to_sentences(output)
            atoms[sentence] = facts

        for key, value in self.demos.items():
            if key not in atoms:
                atoms[key] = list(value)
        return atoms
