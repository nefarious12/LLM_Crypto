# Project 5: How Good Is Your RAG App? An Evaluation

---

My RAG demo gave good-looking answers. "Looking good" is not a measurement.

So I built a test set of 25 questions (21 answerable, 4 not) and scored the pipeline on five metrics. Then I changed settings one at a time. 👇

**Metrics**
- **Retrieval hit:** did the retrieved text contain the facts needed?
- **Answer correct:** does the answer contain them?
- **False abstain:** did it refuse a question it could answer?
- **Correct abstain:** did it refuse a question the report can't answer?
- **Grounded:** do the numbers in the answer appear in the retrieved text?

**Results (120-word chunks, top 3)**
- Retrieval hit: 100%
- Answer correct: 86%
- False abstain: 10%
- Correct abstain: 100%

**What I learned**
1️⃣ **Retrieval was not the problem.** All three failures had the right text retrieved. Two were wrongly refused by the "not found" threshold, and one picked sentences that missed part of the answer. Without separate metrics, I'd have tuned the wrong part.

2️⃣ **Fixing the answer step helped more than I expected.** My first version sometimes returned a section heading as part of the answer. Filtering headings raised answer correctness from 81% to 86% with no change to retrieval.

3️⃣ **Chunk size changed results a lot.** 40-word chunks were 90% correct but refused only half the unanswerable questions. 120-word chunks were 86% correct and refused all of them. 300-word chunks dropped to 29%, because large chunks dilute the similarity score and the system refused most questions.

4️⃣ **The abstention threshold is a trade-off.** At 120 words, lowering it raised correct answers from 86% to 95%, but the system then refused only 50% of the unanswerable questions instead of 100%. More answers means more risk of answering what you shouldn't.

5️⃣ **Some metrics are weak here.** "Grounded" hit 100%, but that's expected for extractive answers. It matters once an LLM writes the text.

**Limits:** 25 questions on one synthetic document is a small test. The method is the point, not the numbers.

**Key takeaway:** Measure retrieval and answers separately, and tune the refusal threshold on purpose.

🔗 Code and test set: https://github.com/nefarious12/Crypto/tree/main/genai-finance

🔜 **Next:** Making it safe. Prompt injection, PII and audit trails.

💬 Which metric would you add?

*Synthetic data. Educational content only, not investment advice.*

#LLMEvaluation #RAG #GenerativeAI #FinTech #Python
