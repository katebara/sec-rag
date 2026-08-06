"""
Hybrid retrieval: combine BM25 keyword search with vector similarity
search, since dense embeddings alone can under-rank exact terminology
(specific dollar figures, section names, technical terms).
"""

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import re

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sec_filings"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

def load_all_chunks(collection):
    """Pull every chunk out of Chroma once, to build the BM25 index."""
    print("Loading all chunks from Chroma for BM25 indexing...")
    data = collection.get(include=["documents", "metadatas"])
    return {
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
    }

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_bm25(documents):
    tokenized = [tokenize(doc) for doc in documents]
    return BM25Okapi(tokenized)

def normalize(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def hybrid_retrieve(query, k=5, alpha=0.5, overfetch=30,
                     embedder=None, collection=None, bm25=None, all_chunks=None):
    """
    alpha: weight given to vector score vs BM25 score.
    alpha=1.0 -> pure vector, alpha=0.0 -> pure BM25.
    """
    # --- vector search (overfetch to give fusion something to work with) ---
    query_embedding = embedder.encode([query])
    vec_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=overfetch,
    )
    vec_ids = vec_results["ids"][0]
    vec_distances = vec_results["distances"][0]
    # Chroma returns distance (lower = better) -> convert to similarity (higher = better)
    vec_scores = {vid: 1 - d for vid, d in zip(vec_ids, vec_distances)}

    # --- BM25 search over the FULL corpus ---
    tokenized_query = tokenize(query)
    bm25_scores_all = bm25.get_scores(tokenized_query)
    bm25_by_id = dict(zip(all_chunks["ids"], bm25_scores_all))

    # --- combine: only fuse over the candidate set vector search returned ---
    candidate_ids = list(vec_scores.keys())
    v_scores = [vec_scores[cid] for cid in candidate_ids]
    b_scores = [bm25_by_id[cid] for cid in candidate_ids]

    v_norm = normalize(v_scores)
    b_norm = normalize(b_scores)

    combined = [
        (cid, alpha * v + (1 - alpha) * b)
        for cid, v, b in zip(candidate_ids, v_norm, b_norm)
    ]
    combined.sort(key=lambda x: x[1], reverse=True)
    top = combined[:k]

    # look up text/metadata for the winners
    id_to_idx = {cid: i for i, cid in enumerate(all_chunks["ids"])}
    results = []
    for cid, score in top:
        idx = id_to_idx[cid]
        results.append({
            "id": cid,
            "score": score,
            "text": all_chunks["documents"][idx],
            "metadata": all_chunks["metadatas"][idx],
        })
    return results

def print_results(results):
    for i, r in enumerate(results):
        m = r["metadata"]
        print(f"\n--- Result {i+1} (hybrid score: {r['score']:.4f}) ---")
        print(f"Company: {m['company']} | FY: {m['fiscal_year']} | Section: {m['section']}")
        print(r["text"][:300].replace("\n", " ") + "...")

if __name__ == "__main__":
    print("Loading embedding model and vector store...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    collection = get_collection()
    all_chunks = load_all_chunks(collection)
    print(f"Loaded {len(all_chunks['ids'])} chunks. Building BM25 index...")
    bm25 = build_bm25(all_chunks["documents"])
    print("Ready.\n")

    while True:
        query = input("Enter a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        results = hybrid_retrieve(
            query, k=5, alpha=0.5,
            embedder=embedder, collection=collection, bm25=bm25, all_chunks=all_chunks
        )
        print_results(results)
        print()
