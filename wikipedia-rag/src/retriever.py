"""
retriever.py — Query classification and retrieval pipeline for Wikipedia RAG.

Classifies queries as "person", "place", or "both", then retrieves relevant
chunks from ChromaDB and enriches them with text from SQLite.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.embedder import embed_text
from src.database import get_chunk_by_chroma_id
from src.vector_store import query_collection

PERSON_KEYWORDS = [
    "who", "born", "died", "discovered", "invented", "wrote",
    "painted", "composed", "founded", "nationality", "biography",
    "early life", "career", "achievement", "award", "scientist",
    "artist", "author", "philosopher", "musician", "athlete"
]

PLACE_KEYWORDS = [
    "where", "located", "location", "city", "country", "continent",
    "built", "construction", "tower", "mountain", "river", "ocean",
    "temple", "monument", "landmark", "height", "size", "area",
    "visit", "travel", "tourism"
]


def classify_query(query: str, people: list[str], places: list[str]) -> str:
    """
    Returns "person", "place", or "both".

    Logic (in order):
    1. Check if any person name appears in query (case-insensitive) → person_name_match = True
    2. Check if any place name appears in query (case-insensitive) → place_name_match = True
    3. If both name matches → return "both"
    4. Count keyword matches for person vs place keywords (lowercase query)
    5. If only person_name_match → "person"
    6. If only place_name_match → "place"
    7. If person_score > place_score → "person"
    8. If place_score > person_score → "place"
    9. Default → "both"
    """
    query_lower = query.lower()

    # Step 1 & 2: Check for name matches (full name or any individual name part)
    def name_in_query(name):
        if name.lower() in query_lower:
            return True
        return any(part.lower() in query_lower for part in name.split() if len(part) > 3)

    person_name_match = any(name_in_query(p) for p in people)
    place_name_match = any(name_in_query(p) for p in places)

    # Step 3: Both name matches → "both"
    if person_name_match and place_name_match:
        return "both"

    # Step 4: Count keyword matches
    person_score = sum(1 for kw in PERSON_KEYWORDS if kw in query_lower)
    place_score = sum(1 for kw in PLACE_KEYWORDS if kw in query_lower)

    # Step 5: Only person name matched
    if person_name_match and not place_name_match:
        return "person"

    # Step 6: Only place name matched
    if place_name_match and not person_name_match:
        return "place"

    # Step 7: Person keywords win
    if person_score > place_score:
        return "person"

    # Step 8: Place keywords win
    if place_score > person_score:
        return "place"

    # Step 9: Default
    return "both"


def retrieve(
    query: str,
    collection,
    conn,
    n_results: int = 5
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Classify query using PEOPLE and PLACES from src.entities
    2. Embed query using embedder.embed_text()
    3. Query ChromaDB with optional type filter
    4. For each result, fetch chunk text from SQLite using get_chunk_by_chroma_id
    5. Return enriched results

    Returns list of:
    {
        "entity_title": str,
        "entity_type": str,
        "chunk_text": str,
        "chunk_index": int,
        "distance": float,
        "query_type": str   # the classification result
    }
    """
    from src.entities import PEOPLE, PLACES

    # Step 1: Classify the query
    query_type = classify_query(query, PEOPLE, PLACES)

    # Step 2: Embed the query
    query_embedding = embed_text(query)
    if query_embedding is None:
        return []

    # Step 3: Query ChromaDB with optional type filter
    filter_type = None if query_type == "both" else query_type
    chroma_results = query_collection(collection, query_embedding, n_results=n_results, filter_type=filter_type)

    # Step 4 & 5: Fetch chunk text from SQLite and build enriched results
    enriched = []
    for result in chroma_results:
        chunk = get_chunk_by_chroma_id(conn, result["chroma_id"])
        chunk_text = chunk["chunk_text"] if chunk is not None else ""
        chunk_index = chunk["chunk_index"] if chunk is not None else result.get("chunk_index", 0)

        enriched.append({
            "entity_title": result["entity_title"],
            "entity_type": result["entity_type"],
            "chunk_text": chunk_text,
            "chunk_index": chunk_index,
            "distance": result["distance"],
            "query_type": query_type,
        })

    return enriched


if __name__ == "__main__":
    from src.entities import PEOPLE, PLACES

    assert classify_query("Who was Albert Einstein?", PEOPLE, PLACES) == "person"
    assert classify_query("Where is the Eiffel Tower?", PEOPLE, PLACES) == "place"
    assert classify_query("Compare Einstein and the Eiffel Tower", PEOPLE, PLACES) == "both"
    assert classify_query("something completely unrelated xyz", PEOPLE, PLACES) == "both"
    print("Classifier tests PASSED")
    print("retriever.py smoke test PASSED")
