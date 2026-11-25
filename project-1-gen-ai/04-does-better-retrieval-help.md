# Bonus Post: Does Fancier Retrieval Help? I Tested It

---

When a RAG app gets answers wrong, the usual advice is "improve your retrieval." Before swapping components, I tested whether retrieval was actually the bottleneck. 👇

**The test**
I compared three retrieval methods on my 21 answerable finance questions:
- TF-IDF on phrases (my original)
- TF-IDF on single words
- BM25, the classic ranking function used in many search engines

A "hit" means the retrieved text contained every fact needed for the answer. I checked it at two chunk sizes.

**Results (120-word chunks)**
| Method | Right chunk ranked first | In top 3 |
|---|---|---|
| TF-IDF phrases | 81% | 100% |
| TF-IDF single words | 81% | 95% |
| BM25 | 81% | 100% |

At 40-word chunks, BM25 ranked the right chunk first for 76% of questions, against 71% for both TF-IDF versions. In top 3, all three methods reached 100% except single-word TF-IDF at 120 words.

**What this says**
✅ The methods are nearly tied. The best gap is one question out of 21.
✅ Top-3 retrieval was already at 95–100%, so there was little left to gain.
✅ This matches the earlier evaluation: my errors came from the answer step and the refusal threshold, not from retrieval.

**What it doesn't say**
- 21 questions on one short document can't separate close methods. One question moves the score by about 5 points.
- On long filings with many similar passages, differences between methods are likely to be larger. Embeddings may also matter more there. I haven't tested that.
- I have not yet compared extractive answers against an LLM. That is the next experiment.

**Key takeaway:** Find the failing stage before you upgrade a component. A ten-minute measurement can save a week of tuning the wrong thing.

🔗 Code: https://github.com/nefarious12/LLM_Crypto/tree/main/project-1-gen-ai/code (`python -m genai_finance.compare_retrievers`)

💬 Have you ever "fixed" a part of a system that turned out not to be the problem?

*Synthetic data. Educational content only, not investment advice.*

#RAG #LLMEvaluation #InformationRetrieval #GenerativeAI #FinTech
