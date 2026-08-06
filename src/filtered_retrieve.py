"""
Add company detection + metadata filtering on top of hybrid search
and reranking, so a query naming a specific company only searches
that company's chunks — preventing cross-company contamination.
"""

import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.hybrid_retrieve import get_collection, load_all_chunks, build_bm25, tokenize, normalize
from src.rerank import rerank, print_results
from rank_bm25 import BM25Okapi

# Map tickers to the names/aliases that might appear in a query.
COMPANY_ALIASES = {
    "NVDA": ["nvidia", "nvda"],
    "AMD": ["amd", "advanced micro devices"],
    "INTC": ["intel", "intc"],
    "QCOM": ["qualcomm", "qcom"],
    "AVGO": ["broadcom", "avgo"],
}

def detect_company(query):
    q_lower = query.lower()
    for ticker, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q_lower):
                return ticker
    return None

def filtered_hybrid_retrieve(query, k=5, alpha=0.5, overfetch=30,
                              embedder=None, collection=None,
                              company_filter=None):
    """
    Same idea as hybrid_retrieve, but pre-filters both vector search
    and BM25 to a single company's chunks when one is detected.
    """
    where = {"company": company_filter} if company_filter else None

    # --- vector search, filtered ---
    query_embedding = embedder.encode([query])
    vec_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=overfetch,
        where=where,
    )
    vec_ids = vec_results["ids"][0]
    vec_docs = vec_results["documents"][0]
    vec_metas = vec_results["metadatas"][0]
    vec_distances = vec_results["distances"][0]
    vec_scores = {vid: 1 - d for vid, d in zip(vec_ids, vec_distances)}

    if not vec_ids:
        return []

    # --- BM25, filtered to the same candidate set's company ---
    # Build a small BM25 index just over the filtered company's chunks
    # (cheaper than rebuilding over all 11k chunks every query).
    if company_filter:
        filtered = collection.get(where=where, include=["documents", "metadatas"])
    else:
        filtered = collection.get(include=["documents", "metadatas"])
    tokenized = [tokenize(doc) for doc in filtered["documents"]]
    local_bm25 = BM25Okapi(tokenized)
    tokenized_query = tokenize(query)
    bm25_scores_all = local_bm25.get_scores(tokenized_query)
    bm25_by_id = dict(zip(filtered["ids"], bm25_scores_all))

    candidate_ids = list(vec_scores.keys())
    v_scores = [vec_scores[cid] for cid in candidate_ids]
    b_scores = [bm25_by_id.get(cid, 0.0) for cid in candidate_ids]
    v_norm = normalize(v_scores)
    b_norm = normalize(b_scores)

    combined = [
        (cid, alpha * v + (1 - alpha) * b)
        for cid, v, b in zip(candidate_ids, v_norm, b_norm)
    ]
    combined.sort(key=lambda x: x[1], reverse=True)

    id_to_doc = dict(zip(vec_ids, vec_docs))
    id_to_meta = dict(zip(vec_ids, vec_metas))

    results = []
    for cid, score in combined[:k]:
        results.append({
            "id": cid,
            "score": score,
            "text": id_to_doc[cid],
            "metadata": id_to_meta[cid],
        })
    return results

if __name__ == "__main__":
    print("Loading models and vector store...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    collection = get_collection()
    print("Ready.\n")

    while True:
        query = input("Enter a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        company = detect_company(query)
        if company:
            print(f"[Detected company filter: {company}]")

        candidates = filtered_hybrid_retrieve(
            query, k=20, alpha=0.5,
            embedder=embedder, collection=collection,
            company_filter=company,
        )
        if not candidates:
            print("No results found.")
            continue

        results = rerank(query, candidates, reranker, top_n=5)
        print_results(results)
        print()
