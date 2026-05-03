"""
ingest.py — Main ingestion script for the Wikipedia RAG pipeline.

Pipeline per entity:
  1. Check SQLite: if already ingested, skip.
  2. Fetch Wikipedia text.
  3. Store entity in SQLite.
  4. Chunk the text.
  5. For each chunk:
       a. Generate chroma_id = f"{title.replace(' ', '_')}_{chunk_index}"
       b. Skip if chroma_id already in ChromaDB.
       c. Embed chunk text.
       d. Store chunk in SQLite.
       e. Store embedding in ChromaDB.
  6. Print progress line.

Idempotent: safe to run multiple times.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.database import get_connection, init_db, entity_exists, insert_entity, insert_chunk
from src.fetcher import fetch_wikipedia_text
from src.chunker import chunk_entity
from src.embedder import check_ollama_running, embed_text
from src.vector_store import get_collection, add_chunk, chunk_already_indexed
from src.entities import PEOPLE, PLACES


def ingest_entity(conn, collection, title: str, entity_type: str, index: int, total: int) -> int:
    """
    Ingest a single entity.  Returns the number of chunks embedded (0 if skipped).
    Raises on unrecoverable errors so the caller can catch and continue.
    """
    prefix = f"[{index}/{total}]"

    # Step 1 — skip if already in SQLite
    if entity_exists(conn, title):
        print(f"{prefix} Skipping {title} ({entity_type}) — already ingested.")
        return 0

    print(f"{prefix} Ingesting {title} ({entity_type})...", end=" ", flush=True)

    # Step 2 — fetch Wikipedia text
    result = fetch_wikipedia_text(title)
    if not result["success"]:
        raise RuntimeError(f"Fetch failed: {result.get('error', 'unknown error')}")

    url = result["url"]
    text = result["text"]

    # Step 3 — store entity in SQLite using the original input title so
    # entity_exists() always matches on re-runs (Wikipedia normalizes titles).
    entity_id = insert_entity(conn, title, entity_type, url, text)

    # Step 4 — chunk the text
    chunks = chunk_entity(title, text)

    chunks_embedded = 0

    # Step 5 — process each chunk
    for chunk in chunks:
        chunk_index = chunk["chunk_index"]
        chunk_text = chunk["chunk_text"]
        token_count = chunk["token_count"]

        # 5a — generate chroma_id
        chroma_id = f"{title.replace(' ', '_')}_{chunk_index}"

        # 5b — skip if already in ChromaDB
        if chunk_already_indexed(collection, chroma_id):
            continue

        # 5c — embed
        embedding = embed_text(chunk_text)
        if embedding is None:
            print(f"\n  Warning: embedding failed for chunk {chunk_index} of '{title}', skipping chunk.")
            continue

        # 5d — store chunk in SQLite
        sqlite_chunk_id = insert_chunk(conn, entity_id, chunk_index, chunk_text, token_count, chroma_id)

        # 5e — store in ChromaDB
        add_chunk(
            collection,
            chroma_id=chroma_id,
            embedding=embedding,
            chunk_text=chunk_text,
            entity_title=title,
            entity_type=entity_type,
            chunk_index=chunk_index,
            sqlite_chunk_id=sqlite_chunk_id,
        )

        chunks_embedded += 1

    # Commit SQLite transaction per entity
    conn.commit()

    print(f"{chunks_embedded} chunks embedded. Done.")
    return chunks_embedded


def main():
    # Pre-flight: verify Ollama is reachable
    if not check_ollama_running():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Ensure the data directory and DB schema exist
    os.makedirs("data", exist_ok=True)
    init_db()

    conn = get_connection()
    collection = get_collection()

    entities = (
        [(name, "person") for name in PEOPLE] +
        [(name, "place")  for name in PLACES]
    )
    total = len(entities)

    total_entities = 0
    total_chunks = 0

    for idx, (title, entity_type) in enumerate(entities, start=1):
        try:
            chunks_embedded = ingest_entity(conn, collection, title, entity_type, idx, total)
            if chunks_embedded > 0:
                total_entities += 1
                total_chunks += chunks_embedded
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR ingesting '{title}': {exc}")

    conn.close()
    print(f"\nIngestion complete. {total_entities} entities, {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
