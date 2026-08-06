"""
Add cross-encoder reranking on top of hybrid retrieval. The reranker
reads the query and each candidate chunk TOGETHER, so it can catch
cases where keyword/vector overlap doesn't mean true relevance
(e.g. an INTC chunk matching "AMD R&D spending" on keywords alone).
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from src.hybrid_retrieve import get_collection, load_all_chunks, build_bm25, hybrid_retrieve

def rerank(query, candidates, reranker, top_n=5):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_n]]

def print_results(results, score_key="rerank_score"):
    for i, r in enumerate(results):
        m = r["metadata"]
        print(f"\n--- Result {i+1} ({score_key}: {r[score_key]:.4f}) ---")
        print(f"Company: {m['company']} | FY: {m['fiscal_year']} | Section: {m['section']}")
        print(r["text"][:300].replace("\n", " ") + "...")

if __name__ == "__main__":
    print("Loading models and vector store...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    collection = get_collection()
    all_chunks = load_all_chunks(collection)
    bm25 = build_bm25(all_chunks["documents"])
    print("Ready.\n")

    while True:
        query = input("Enter a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        # overfetch with hybrid search, then let the reranker narrow it down
        candidates = hybrid_retrieve(
            query, k=20, alpha=0.5,
            embedder=embedder, collection=collection, bm25=bm25, all_chunks=all_chunks
        )
        results = rerank(query, candidates, reranker, top_n=5)
        print_results(results)
        print()
