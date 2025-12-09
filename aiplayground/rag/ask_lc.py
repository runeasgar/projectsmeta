# LangChain ask: retrieve (Qdrant) → prompt → answer (ChatOllama)
# Static config by design to keep the lab simple.

import os, textwrap
from typing import List, Tuple

from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ---------- CONFIG (edit here) ----------
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "rag_langchain_v1"          # created in LC-1 and populated in LC-2
EMBED_MODEL  = "nomic-embed-text"          # pulled via `ollama pull nomic-embed-text`
OLLAMA_URL   = "http://127.0.0.1:11434"
LLM_MODEL    = "llama3.2:3b-instruct-q4_K_M"
TOP_K        = 3                           # small on purpose to feel ranking
SNIPPET_CHARS= 1000                        # give the model fuller context
TEMPERATURE  = 0.2                         # low for grounded RAG

# ---------- HELPERS ----------
def format_sources(docs_with_scores) -> Tuple[str, list[Tuple[int, str, float]]]:
    """
    Create a numbered source block and a legend.
    Returns:
      context_block: string to feed the LLM
      legend: [(n, key, score)] for terminal display
    """
    lines, legend = [], []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        idx = doc.metadata.get("chunk_index", "?")
        key = f"{src}#{idx}"
        txt = (doc.page_content or "").replace("\n", " ")
        if len(txt) > SNIPPET_CHARS:
            txt = txt[:SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"[{i}] {key}: {txt}")
        legend.append((i, key, float(score)))
    return "\n".join(lines), legend

# ---------- MAIN ----------
def main():
    # init embeddings & vector store (LangChain wrappers)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    client = QdrantClient(url=QDRANT_URL)
    vs = Qdrant(
        client=client,
        collection_name=COLLECTION,
        embeddings=embeddings,
        content_payload_key="text",
    )

    # get the question
    try:
        question = input("Question: ").strip()
    except KeyboardInterrupt:
        return
    if not question:
        print("No question provided.")
        return

    # retrieve with scores (uses embeddings.embed_query under the hood)
    docs_with_scores = vs.similarity_search_with_score(question, k=TOP_K)
    if not docs_with_scores:
        print("No results retrieved from Qdrant.")
        return

    # show ranked results and build the context we’ll send to the LLM
    context_block, legend = format_sources(docs_with_scores)

    print("\n--- Retrieved (ranked) ---")
    for n, key, score in legend:
        print(f"[{n}] score={score:.4f}  {key}")

    print("\n--- Sources sent to LLM ---\n" + context_block)

    # strict prompt template (LangChain): system + human
    system_text = (
        "Answer ONLY using the provided sources.\n"
        "Every factual sentence MUST end with one or more bracketed citations like [1] or [2][3].\n"
        "If the sources do not contain the answer, say you don't know.\n"
        "Do NOT invent citations."
    )
    user_template = textwrap.dedent("""
        Question: {question}

        Sources:
        {sources}
    """).strip()

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", user_template)]
    )

    # ChatOllama (LangChain chat model wrapper around your local Ollama)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)

    # Compose & run: prompt → llm
    chain = prompt | llm
    ai_msg = chain.invoke({"question": question, "sources": context_block})

    print("\n--- Answer ---\n" + (ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)))

    print("\n--- Sources legend ---")
    for n, key, score in legend:
        print(f"[{n}] {key}  (score={score:.4f})")

if __name__ == "__main__":
    main()
