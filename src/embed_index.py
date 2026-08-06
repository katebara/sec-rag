"""
Chunk processed filing sections, embed them, and store in a
persistent Chroma vector database.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROCESSED_DIR = Path("data/processed")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "sec_filings"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # characters of overlap between chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_all_filings():
    filings = []
    for filepath in PROCESSED_DIR.glob("*.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            filings.append(json.load(f))
    return filings

def build_chunks(filings):
    """Turn all filings' sections into a flat list of chunk records."""
    all_chunks = []
    for filing in filings:
        company = filing["company"]
        fiscal_year = filing["fiscal_year"]
        for section_name, section_text in filing["sections"].items():
            if not section_text or len(section_text.strip()) < 20:
                continue  # skip empty/near-empty sections
            pieces = chunk_text(section_text)
            for i, piece in enumerate(pieces):
                chunk_id = f"{company}_{fiscal_year}_{section_name}_{i}".replace(" ", "_").replace("/", "-")
                all_chunks.append({
                    "id": chunk_id,
                    "text": piece,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "section": section_name,
                })
    return all_chunks

def build_index(chunks, batch_size=100):
    print("Loading embedding model...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # start fresh each run to avoid duplicate/stale entries
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {"company": c["company"], "fiscal_year": c["fiscal_year"], "section": c["section"]}
                for c in batch
            ],
        )
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")

if __name__ == "__main__":
    filings = load_all_filings()
    print(f"Loaded {len(filings)} filings")

    chunks = build_chunks(filings)
    print(f"Built {len(chunks)} chunks total")

    build_index(chunks)
