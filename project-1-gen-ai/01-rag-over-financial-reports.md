# Project 1: RAG over Financial Reports

---

Ask an LLM about a company's annual report and it may answer with confidence from memory, not from the document.

For finance, "probably right" isn't good enough. I built a small RAG pipeline where every answer must point to a passage, and the system says "not found" when the report doesn't contain the answer. 👇

**How it works**
1️⃣ **Chunk:** split the report into passages of about 120 words, keeping paragraphs intact.
2️⃣ **Index:** convert each chunk into a searchable representation (TF-IDF here, embeddings for real documents).
3️⃣ **Retrieve:** find the top 3 chunks for the question.
4️⃣ **Answer:** return the best-matching sentences with chunk citations, or use an LLM restricted to those chunks.
5️⃣ **Abstain:** if the best match scores below a threshold, reply "I could not find this in the provided documents."

**Example**
Q: What was total revenue in fiscal 2024?
A: Total revenue for fiscal 2024 was $4.82 billion, an increase of 9% from $4.42 billion in fiscal 2023. [1] (shortened: the app also returns a second supporting sentence)

Q: What is the CEO's annual salary?
A: I could not find this in the provided documents. ✅ (the report doesn't say)

**Design choices that matter in finance**
- Citations on every answer, so a reviewer can check the source in seconds
- Abstaining over guessing
- The LLM mode is told to treat retrieved text as data, never as instructions
- It runs offline with no API key, so anyone can reproduce it

**Honest limits**
The demo uses a synthetic company report and keyword-based retrieval. It's a working skeleton, not a production system. The next post measures how well it actually performs.

**Key takeaway:** In finance, a RAG system is only as good as its citations and its willingness to say "I don't know."

🔗 Code: https://github.com/nefarious12/Crypto/tree/main/genai-finance

🔜 **Next:** How do you know your RAG app is any good? Building an evaluation.

💬 What would you want a finance AI assistant to refuse to answer?

*Synthetic data used for demonstration. Educational content only, not investment advice.*

#RAG #GenerativeAI #LLM #FinTech #Python
