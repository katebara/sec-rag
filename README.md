
# SEC 10-K RAG System

A retrieval-augmented Q&A system over SEC 10-K filings for five semiconductor companies — NVIDIA (NVDA), AMD, Intel (INTC), Qualcomm (QCOM), and Broadcom (AVGO) — across fiscal years 2022–2025.

Ask natural-language questions like *"How did AMD's R&D spending change year over year?"* or *"Compare NVIDIA and Qualcomm's stated risk factors around China"* and get answers grounded in the actual filing text, with citations back to the source section — not a model's guess from memory.

## Why this project

General-purpose LLMs don't reliably know specific, current financial figures, and they'll confidently hallucinate an answer rather than say "I don't know." This project builds a retrieval pipeline that grounds every answer in real filing text, and — just as importantly — measures how well that grounding actually works, rather than assuming it does. SEC filings are a good stress test for this: they're long, densely structured, and (as this project found) far less uniformly formatted across companies than their standardized "Item" numbering suggests.

## What it does

1. **Ingests** 10-K filings directly from SEC EDGAR (no scraping, official free API)
2. **Parses** each filing's SGML/Inline-XBRL wrapper into clean text, then splits it into its standardized sections (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A, Item 8 Financials, etc.) — with a fallback to generic chunking for filers whose headers aren't machine-matchable as text (see `EXPERIMENTS.md`)
3. **Chunks and embeds** sections into a vector store, with metadata (company, fiscal year, section) enabling filtered retrieval
4. **Retrieves** relevant chunks per query using hybrid search — dense vector similarity plus BM25 keyword matching — since financial text often hinges on exact terms (ticker symbols, statute numbers, specific line items) that pure semantic search can miss
5. **Reranks** the top candidates with a cross-encoder before generation, to sharpen precision beyond what first-pass retrieval alone achieves
6. **Generates** answers grounded only in retrieved text, with inline citations back to source chunks
7. **Evaluates** the whole pipeline against a hand-built question set with verifiable answers — retrieval recall, answer faithfulness, and exact-match accuracy on numeric facts — rather than relying on "the answer sounded right"

## Architecture

EDGAR download -> SGML/XBRL extraction -> section-aware parsing (+ fallback)
-> chunking -> embedding (BGE-small) -> vector store (Chroma) + BM25 index
-> hybrid retrieval -> cross-encoder reranking -> LLM generation with citations
-> evaluation harness (retrieval recall, faithfulness, numeric accuracy)

## Status

- [x] Environment & repo setup
- [x] EDGAR ingestion (all 5 companies, FY2022–2025)
- [x] Section-aware parsing with cross-filer fallback handling
- [ ] Chunking + embedding + vector store
- [ ] Hybrid retrieval (dense + BM25)
- [ ] Cross-encoder reranking
- [ ] Hand-built evaluation set (30–50 verifiable Q&A pairs)
- [ ] Evaluation harness (recall@k, faithfulness, numeric accuracy)
- [ ] Demo app (Streamlit)

## Key engineering decisions worth noting

- **Not every SEC filer formats section headers the same way.** NVDA, AMD, QCOM, and AVGO all embed matchable "Item N." text in-body, but with different casing, spacing, and line-break conventions. Intel's rendering doesn't include matchable header text in-body at all — the pipeline detects this per-filing and falls back to generic chunking rather than silently producing broken output. Full debugging trail in `EXPERIMENTS.md`.
- **Hybrid retrieval over pure vector search** — financial questions often hinge on exact terms (specific dollar figures, section numbers, company names) that dense embeddings alone can under-retrieve.
- **Evaluation is numeric, not just vibes-based** — because 10-Ks contain objectively verifiable facts, this project measures exact-match accuracy on retrieved numbers, not just whether an answer "sounds" plausible.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your own SEC_USER_NAME / SEC_USER_EMAIL
python src/ingest.py
python src/parse.py
```

## Data source

All filings pulled directly from [SEC EDGAR](https://www.sec.gov/edgar), free and public.