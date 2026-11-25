"""Compare retrieval methods offline, independent of any answer generator.

Methods: TF-IDF (1-2 word phrases), TF-IDF (single words), BM25.
Metrics on the 21 answerable questions, a hit meaning a retrieved chunk contains ALL expected facts:
  hit@1, hit@3   fraction of questions with a hit in the top 1 / top 3 chunks
  MRR            mean of 1 / rank of the first hit (0 if none in the top 5)

Run: python -m genai_finance.compare_retrievers
Educational only. Not investment advice.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .evaluate import load_eval_set, norm
from .rag import DATA_DIR, Chunk, Retriever, build_chunks

TOKEN = re.compile(r"[a-z0-9$%.,]+")


def tokenize(text: str) -> list[str]:
    toks = [t.strip(".,") for t in TOKEN.findall(text.lower())]
    return [t for t in toks if t and t not in ENGLISH_STOP_WORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks))
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(t, 0)
                if tf:
                    out[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def ranker_tfidf(chunks, ngram):
    vec = TfidfVectorizer(ngram_range=ngram, stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform([c.text for c in chunks])
    return lambda q: (mat @ vec.transform([q]).T).toarray().ravel()


def evaluate_ranker(chunks, scorer, k_max=5) -> dict:
    qs = [q for q in load_eval_set() if q["answerable"]]
    h1 = h3 = 0
    rr = []
    for q in qs:
        facts = [norm(e) for e in q["expected"]]
        order = np.argsort(-scorer(q["question"]))[:k_max]
        rank = next((r for r, i in enumerate(order, 1) if all(f in norm(chunks[i].text) for f in facts)), None)
        h1 += rank == 1
        h3 += rank is not None and rank <= 3
        rr.append(1 / rank if rank else 0.0)
    n = len(qs)
    return {"hit@1": h1 / n, "hit@3": h3 / n, "MRR": float(np.mean(rr))}


def run() -> list[dict]:
    rows = []
    for words in (40, 120):
        chunks = build_chunks([DATA_DIR / "meridian_fy2024.txt"], max_words=words, overlap=min(20, words // 4))
        methods = {
            "TF-IDF 1-2 words": ranker_tfidf(chunks, (1, 2)),
            "TF-IDF 1 word": ranker_tfidf(chunks, (1, 1)),
            "BM25": BM25(chunks).scores,
        }
        for name, scorer in methods.items():
            rows.append({"chunk_words": words, "method": name, "chunks": len(chunks), **evaluate_ranker(chunks, scorer)})
    return rows


if __name__ == "__main__":
    print(f"{'words':>5} {'chunks':>6}  {'method':<18}{'hit@1':>7}{'hit@3':>7}{'MRR':>7}")
    for r in run():
        print(f"{r['chunk_words']:>5} {r['chunks']:>6}  {r['method']:<18}{r['hit@1']:>7.0%}{r['hit@3']:>7.0%}{r['MRR']:>7.2f}")
