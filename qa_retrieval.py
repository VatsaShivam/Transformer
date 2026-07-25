"""
Closed-domain question answering over the lighthouse story from
nlp_char_transformer.py.

Why retrieval, not the generative char-model: the char-level transformer
we trained has no concept of "question" vs "answer" -- it was only ever
trained to predict the next character of the story. Prompting it with a
question just makes it continue in story-voice, drifting into memorized
fragments (as the previous test run showed). That's the wrong tool for
this job.

What actually answers questions reliably about *this specific text* is
classic extractive QA: split the text into sentences, represent both the
sentences and the question as TF-IDF vectors, and return the sentence(s)
most similar to the question by cosine similarity. This is grounded --
every answer is an exact sentence from the source text, so it can't
hallucinate content that isn't there -- and it's the standard approach
for document QA before generative answering became viable.

Run with:
    python3 qa_retrieval.py
"""

import re
import numpy as np

from nlp_char_transformer import CORPUS


# ==========================================================================
# 1. Split the story into sentences
# ==========================================================================

def split_sentences(text: str):
    """Split on sentence-ending punctuation, keeping the punctuation."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ==========================================================================
# 2. Tokenize into words (lowercased, punctuation stripped)
# ==========================================================================

_WORD_RE = re.compile(r"[a-z']+")

# Function words carry almost no topical meaning but, in a corpus this
# small, can accidentally get a high IDF (they appear in very few
# sentences purely by chance) and end up dominating the similarity score
# for short questions. Filtering them out keeps scoring focused on
# content words (nouns, verbs, adjectives) that actually indicate topic.
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "and", "or", "but",
    "is", "was", "were", "are", "be", "been", "that", "who", "which",
    "she", "he", "they", "her", "him", "them", "it", "its", "this",
    "these", "those", "for", "with", "as", "so", "than", "then", "when",
    "where", "how", "what", "why", "did", "do", "does", "doing", "not",
    "never", "no", "yes", "all", "every", "each", "before", "after",
    "during", "while", "from", "into", "up", "down", "out", "over",
    "under", "again", "further", "once", "here", "there", "own", "same",
    "too", "very", "can", "will", "just", "should", "now", "had", "has",
    "have", "having", "would", "could", "i", "you", "your", "my", "me",
    "we", "our", "us",
}


def tokenize(text: str):
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


# ==========================================================================
# 3. TF-IDF index over the sentences
# ==========================================================================

class TfidfIndex:
    """A minimal TF-IDF index: fit on a list of sentences, then score a
    query against every sentence via cosine similarity."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.tokenized = [tokenize(s) for s in sentences]

        vocab = sorted(set(w for toks in self.tokenized for w in toks))
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        n_docs = len(sentences)

        # Document frequency -> inverse document frequency.
        df = np.zeros(len(vocab))
        for toks in self.tokenized:
            for w in set(toks):
                df[self.word_to_idx[w]] += 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0  # smoothed idf

        # TF-IDF matrix for all sentences, L2-normalized per row.
        self.matrix = np.stack([self._vectorize(toks) for toks in self.tokenized])

    def _vectorize(self, tokens):
        vec = np.zeros(len(self.vocab))
        for w in tokens:
            idx = self.word_to_idx.get(w)
            if idx is not None:
                vec[idx] += 1.0
        vec *= self.idf  # term frequency * inverse document frequency
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def query(self, question: str, top_k: int = 2):
        """Return the top_k (sentence, score) pairs most similar to the question."""
        q_tokens = tokenize(question)
        q_vec = self._vectorize(q_tokens)
        scores = self.matrix @ q_vec  # cosine similarity (rows already normalized)
        order = np.argsort(-scores)[:top_k]
        return [(self.sentences[i], float(scores[i])) for i in order]

    def answer(self, question: str, min_score: float = 0.05):
        """Return the single best-matching sentence, or None if nothing
        in the text is relevant enough to the question."""
        results = self.query(question, top_k=1)
        sentence, score = results[0]
        if score < min_score:
            return None, score
        return sentence, score


# ==========================================================================
# 4. Demo: build the index and answer a range of questions
# ==========================================================================

def main():
    sentences = split_sentences(CORPUS)
    index = TfidfIndex(sentences)

    print(f"Indexed {len(sentences)} sentences, vocabulary of {len(index.vocab)} words.\n")

    questions = [
        "Who tended the lighthouse lamp?",
        "How many years did Elena tend the lamp?",
        "What did the villagers give Elena every autumn?",
        "How did Elena know a storm was coming?",
        "What did children do when they visited?",
        "Why did the village build an elevator?",
        "Did Elena ever move into town?",
        "What is the capital of France?",  # not answerable from this text
    ]

    for q in questions:
        answer, score = index.answer(q)
        print(f'Q: {q}')
        if answer is None:
            print(f"A: (not found in the story -- best match too weak, score={score:.3f})\n")
        else:
            print(f"A: {answer}")
            print(f"   (similarity score: {score:.3f})\n")


if __name__ == "__main__":
    main()
