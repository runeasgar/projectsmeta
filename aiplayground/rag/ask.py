import os, textwrap
from typing import List, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from ollama import Client as OllamaClient

# ---------- config ----------
load_dotenv()
QDRANT_URL  = os.getenv("QDRANT_URL")
OLLAMA_URL  = os.getenv("OLLAMA_URL")
COLLECTION  = "rag_press_release"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2:3b-instruct-q4_K_M"
TOP_K       = int("3")
SNIPPET_CHARS = int("1000")

def embed(ollama: OllamaClient, text: str) -> List[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def retrieve(qdrant: QdrantClient, query_vec: List[float], k: int):
    # Using search() for compatibility with your client version
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

def build_context(points) -> Tuple[str, list[tuple[int, str, float, str]]]:
    """
    Returns:
      context_block (str): numbered snippets to feed the LLM
      legend (list): [(n, key, score, preview_text)]
    """
    lines, legend = [], []
    for i, p in enumerate(points, start=1):
        pl = p.payload or {}
        key = f"{pl.get('source','?')}#{pl.get('chunk_index','?')}"
        txt = (pl.get("text") or "").replace("\n", " ")
        preview = txt if len(txt) <= SNIPPET_CHARS else txt[:SNIPPET_CHARS].rstrip() + "…"
        score = float(getattr(p, "score", 0.0))
        lines.append(f"[{i}] {key}: {preview}")
        legend.append((i, key, score, preview))
    return "\n".join(lines), legend

def main():
    # --- init clients ---
    qdrant = QdrantClient(url=QDRANT_URL)
    ollama = OllamaClient(host=OLLAMA_URL)

    # --- get a question ---
    try:
        question = input("Question: ").strip()
    except KeyboardInterrupt:
        return
    if not question:
        print("No question provided.")
        return

    # --- embed question & retrieve ---
    qvec = embed(ollama, question)
    points = retrieve(qdrant, qvec, TOP_K)
    if not points:
        print("No results retrieved from Qdrant.")
        return

    context_block, legend = build_context(points)

    # --- show what we retrieved (verbosity you asked for) ---
    print("\n--- Retrieved (ranked) ---")
    for n, key, score, prev in legend:
        print(f"[{n}] score={score:.4f}  {key}")
    print("\n--- Sources sent to LLM ---\n" + context_block)

    # --- strict, simple prompt with hard citation rule ---
    system_msg = (
        "You must answer using only the provided sources.\n"
        "If the answer is strongly implied but not stated verbatim, infer it cautiously and cite the supporting snippets.\n"
        "If truly absent, say you don't know.\n"
        "Every factual sentence MUST end with one or more bracketed citations like [1] or [2][3].\n"
        "If the sources do not contain the answer, say you don't know.\n"
        "Do NOT invent citations."
    )
    user_msg = textwrap.dedent(f"""
        Question: {question}

        Sources:
        {context_block}
    """).strip()

    # --- call LLM ---
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2},
    )
    answer = resp["message"]["content"]

    print("\n--- Answer ---\n" + answer)

    # --- legend so [n] means something after the fact ---
    print("\n--- Sources legend ---")
    for n, key, score, prev in legend:
        print(f"[{n}] {key}  (score={score:.4f})")

if __name__ == "__main__":
    main()