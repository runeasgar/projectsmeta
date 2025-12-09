import os, math, pathlib
from typing import List, Dict
from qdrant_client import QdrantClient, models
from ollama import Client as OllamaClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- config ----
load_dotenv()  # loads .env in this folder if present
QDRANT_URL   = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION   = "rag_press_release"
EMBED_MODEL  = "nomic-embed-text"
SOURCE_PATH  = "press-release.txt"
SOURCE_NAME  = pathlib.Path(SOURCE_PATH).name

# Chunking heuristic: ~3000 chars with ~400 overlap ≈ ~800-token-ish chunks (good enough for MVP)
CHUNK_SIZE   = int("3000")
CHUNK_OVERLAP= int("400")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def make_chunks(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_text(text)
    return docs

def main():
    # Clients
    qdrant = QdrantClient(url=QDRANT_URL)
    ollama = OllamaClient(host=OLLAMA_URL)

    # Read & chunk
    text = read_text(SOURCE_PATH)
    chunks = make_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Read {len(text):,} chars → {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Embed each chunk and prepare points
    points = []
    for idx, chunk in enumerate(chunks):
        emb_resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
        vec = emb_resp["embedding"]  # list[float], length must match collection vector size
        points.append(
            models.PointStruct(
                id=idx,  # stable per source; re-running will upsert same ids
                vector=vec,
                payload={
                    "source": SOURCE_NAME,
                    "chunk_index": idx,
                    "text": chunk
                }
            )
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
            print(f" embedded {idx+1}/{len(chunks)}")

    # Upsert to Qdrant
    qdrant.upsert(collection_name=COLLECTION, points=points)
    info = qdrant.get_collection(COLLECTION)
    print(f"Upserted {len(points)} points → collection status={info.status}, vectors_count={getattr(info, 'vectors_count', 'n/a')}")

    # Light verification: fetch 1 nearest to a simple query
    test_query = "What did NASA announce?"
    q = ollama.embeddings(model=EMBED_MODEL, prompt=test_query)["embedding"]
    search = qdrant.search(collection_name=COLLECTION, query_vector=q, limit=3, with_payload=True)
    print("\nTop-3 preview:")
    for r in search:
        meta = r.payload or {}
        print(f"- score={r.score:.4f}  [{meta.get('source')}#{meta.get('chunk_index')}]  {meta.get('text','')[:120].replace('\\n',' ')}...")

if __name__ == "__main__":
    main()
