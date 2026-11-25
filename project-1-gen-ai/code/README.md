# GenAI for Finance: RAG, evaluation and guardrails

Three connected projects on a synthetic annual report. Everything runs offline with no API key. Educational only, not investment advice.

| Project | Module | What it does |
|---|---|---|
| 1. RAG | `genai_finance/rag.py` | Chunk, index (TF-IDF), retrieve, answer with citations, abstain when unsupported |
| 5. Evaluation | `genai_finance/evaluate.py` | Scores the pipeline on 25 labelled questions and sweeps chunk size, top_k and abstention threshold |
| 6. Guardrails | `genai_finance/guardrails.py`, `redteam.py` | Injection checks, PII redaction, quarantine of untrusted documents, audit log, approval gate, red-team tests |

## Run
```
pip install -r requirements.txt
python -m genai_finance.rag "What was total revenue in fiscal 2024?"
python -m genai_finance.evaluate
python -m genai_finance.redteam
python -m genai_finance.compare_retrievers   # TF-IDF vs BM25, retrieval only
```

## Use an LLM instead of extractive answers
Set `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL`. The pipeline then generates answers from the retrieved chunks with any OpenAI-compatible API. Example for Gemini (key from https://aistudio.google.com/apikey):

```
OPENAI_API_KEY=<your Gemini key>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=<a current Gemini model name>
```

Then compare extractive and LLM answers on the 25-question set (about 20 API calls):

```
python -m genai_finance.evaluate --llm
```

Never commit your key. The request code retries on rate limits (HTTP 429). The retry and request path were tested against a local fake server, not yet against a real model.

## Data
`data/meridian_fy2024.txt` is an invented company report. `data/poisoned_doc.txt` is a fake vendor newsletter with a hidden instruction, used to test indirect prompt injection. `data/eval_set.json` holds the 25 test questions (21 answerable, 4 not).

## Limits
- 25 questions on one short document is a small test. Treat the numbers as a demonstration of the method, not a benchmark.
- TF-IDF matches words, not meaning. Swap in an embedding model for real documents.
- Guardrails here use pattern matching. `redteam.py` lists attacks it misses (obfuscation, other languages, spelled-out numbers).


