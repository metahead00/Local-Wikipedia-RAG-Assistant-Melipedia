import os
import chromadb


def get_collection(persist_dir: str = "chroma_store") -> chromadb.Collection:
    """
    Initialize ChromaDB persistent client.
    Get or create collection named "wikipedia_rag".
    Return the collection object.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="wikipedia_rag",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunk(
    collection,
    chroma_id: str,
    embedding: list[float],
    chunk_text: str,
    entity_title: str,
    entity_type: str,   # "person" or "place"
    chunk_index: int,
    sqlite_chunk_id: int
) -> None:
    """Add a single chunk embedding to ChromaDB."""
    collection.add(
        ids=[chroma_id],
        embeddings=[embedding],
        documents=[chunk_text],
        metadatas=[{
            "entity_title": entity_title,
            "type": entity_type,
            "chunk_index": chunk_index,
            "sqlite_chunk_id": sqlite_chunk_id,
        }]
    )


def query_collection(
    collection,
    query_embedding: list[float],
    n_results: int = 5,
    filter_type: str | None = None   # "person", "place", or None for both
) -> list[dict]:
    """
    Query ChromaDB for nearest neighbors.
    If filter_type is not None, apply: where={"type": filter_type}

    Returns list of:
    {
        "chroma_id": str,
        "entity_title": str,
        "entity_type": str,
        "chunk_index": int,
        "sqlite_chunk_id": int,
        "distance": float
    }
    """
    # Cap n_results to the actual collection size to avoid errors
    total = collection_count(collection)
    if total == 0:
        return []
    effective_n = min(n_results, total)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": effective_n,
        "include": ["metadatas", "distances"],
    }
    if filter_type is not None:
        kwargs["where"] = {"type": filter_type}

    try:
        results = collection.query(**kwargs)
    except Exception:
        return []

    output = []
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chroma_id, meta, distance in zip(ids, metadatas, distances):
        output.append({
            "chroma_id": chroma_id,
            "entity_title": meta.get("entity_title", ""),
            "entity_type": meta.get("type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "sqlite_chunk_id": meta.get("sqlite_chunk_id", 0),
            "distance": distance,
        })

    return output


def collection_count(collection) -> int:
    """Return total number of documents in collection."""
    return collection.count()


def chunk_already_indexed(collection, chroma_id: str) -> bool:
    """Return True if chroma_id already exists in collection."""
    result = collection.get(ids=[chroma_id], include=[])
    return len(result["ids"]) > 0


if __name__ == "__main__":
    import shutil
    test_dir = "chroma_store_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    col = get_collection(persist_dir=test_dir)
    fake_embedding = [0.1] * 768

    add_chunk(col, "test-001", fake_embedding, "Einstein was a physicist",
              "Albert Einstein", "person", 0, 1)

    print(f"Collection count: {collection_count(col)}")
    assert collection_count(col) == 1
    assert chunk_already_indexed(col, "test-001") == True
    assert chunk_already_indexed(col, "test-999") == False

    results = query_collection(col, fake_embedding, n_results=1)
    print(f"Query result: {results[0]['entity_title']}")
    assert results[0]["entity_title"] == "Albert Einstein"

    shutil.rmtree(test_dir)
    print("vector_store.py smoke test PASSED")
