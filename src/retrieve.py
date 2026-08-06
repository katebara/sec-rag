"""
Query the Chroma vector store and inspect what comes back, before
adding hybrid search or reranking on top.
"""

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sec_filings"

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

def retrieve(query, k=5, embedder=None, collection=None):
    if embedder is None:
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if collection is None:
        collection = get_collection()

    query_embedding = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
    )
    return results

def print_results(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        print(f"\n--- Result {i+1} (distance: {dist:.4f}) ---")
        print(f"Company: {meta['company']} | FY: {meta['fiscal_year']} | Section: {meta['section']}")
        print(doc[:300].replace("\n", " ") + "...")

if __name__ == "__main__":
    print("Loading embedding model and vector store...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    collection = get_collection()
    print(f"Collection has {collection.count()} chunks. Ready.\n")

    while True:
        query = input("Enter a question (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        results = retrieve(query, k=5, embedder=embedder, collection=collection)
        print_results(results)
        print()
