# LangChain ingestion: load → split → embed (Ollama) → upsert (Qdrant)
# Static config on purpose (keep it simple for the lab).

import os
from collections import defaultdict
from typing import List

from qdrant_client import QdrantClient
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant

# ---------- CONFIG (edit here) ----------
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "rag_langchain_v1"            # you already created this
DATA_FILES   = [
    "./press-release.txt",
    "./interview.txt",
]
EMBED_MODEL  = "nomic-embed-text"            # pulled via `ollama pull nomic-embed-text`
OLLAMA_URL   = "http://127.0.0.1:11434"      # Ollama base URL
CHUNK_SIZE   = 1000                           # 900–1200 is a nice range
CHUNK_OVERLAP= 150

# ---------- PIPELINE ----------
def load_files(paths: List[str]):
    docs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        # TextLoader preserves metadata["source"] = path
        docs.extend(TextLoader(p, autodetect_encoding=True).load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

def add_chunk_index_per_source(chunks):
    # Annotate each chunk with a per-file chunk_index for clean citations later.
    per_source_counter = defaultdict(int)
    for d in chunks:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        idx = per_source_counter[src]
        d.metadata["source"] = src
        d.metadata["chunk_index"] = idx
        per_source_counter[src] += 1
    return chunks

def main():
    print("Loading files...")
    base_docs = load_files(DATA_FILES)
    print(f"Loaded {len(base_docs)} file-level documents")

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_docs(base_docs)
    chunks = add_chunk_index_per_source(chunks)
    print(f"Produced {len(chunks)} chunks total")

    print("Preparing embeddings (Ollama)…")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    print("Connecting to Qdrant and upserting…")
    client = QdrantClient(url=QDRANT_URL)

    # Create a VectorStore bound to the existing collection, then add chunks.
    vs = Qdrant(
        client=client,
        collection_name=COLLECTION,
        embeddings=embeddings,
        content_payload_key="text",   # how LC stores the text in Qdrant payload
    )

    # Upsert all chunks (VectorStore handles batching/embedding calls).
    vs.add_documents(chunks)

    # Light summary by source
    by_src = defaultdict(int)
    for d in chunks:
        by_src[d.metadata["source"]] += 1
    print("Upsert complete.")
    for src, n in by_src.items():
        print(f"  - {src}: {n} chunks")

if __name__ == "__main__":
    main()
