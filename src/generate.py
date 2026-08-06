"""
Generate a grounded answer from retrieved chunks, using a local
LLM (via Ollama). Instructs the model to answer ONLY from the
provided context and cite sources - a single retrieve-then-generate
pass, not an agent.
"""

import requests
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.hybrid_retrieve import get_collection
from src.filtered_retrieve import detect_company, filtered_hybrid_retrieve
from src.rerank import rerank

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[{i}] (Company: {c['metadata']['company']}, FY: {c['metadata']['fiscal_year']}, "
        f"Section: {c['metadata']['section']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""Answer the question using ONLY the context below. Cite sources using [number] notation after each claim. If the context doesn't contain the answer, say so explicitly rather than guessing.

Context:
{context}

Question: {query}

Answer:"""

def call_ollama(prompt, model=MODEL):
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"]

def answer_question(query, embedder, collection, reranker, k=5):
    company = detect_company(query)
    candidates = filtered_hybrid_retrieve(
        query, k=20, alpha=0.5,
        embedder=embedder, collection=collection,
        company_filter=company,
    )
    if not candidates:
        return "No relevant context found.", []

    top_chunks = rerank(query, candidates, reranker, top_n=k)
    prompt = build_prompt(query, top_chunks)
    answer = call_ollama(prompt)
    return answer, top_chunks

if __name__ == "__main__":
    print("Loading models and vector store...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    collection = get_collection()
    print("Ready. (Make sure ollama serve is running in another terminal.)\n")

    while True:
        query = input("Enter a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        answer, sources = answer_question(query, embedder, collection, reranker)
        print(f"\nAnswer:\n{answer}\n")
        print("Sources used:")
        for i, c in enumerate(sources):
            m = c["metadata"]
            print(f"  [{i}] {m['company']} FY{m['fiscal_year']} - {m['section']}")
        print()
