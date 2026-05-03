# TASKS.md — Claude Code Build & QA Agent Workflow
## Local Wikipedia RAG System — BLG 483E Project 3

---

## How to Use This File

This file is structured for a two-agent Claude Code workflow:

- **Builder Agent**: Implements each task exactly as specified
- **QA Agent**: After each task, verifies correctness using the QA checklist before proceeding

**Rules:**
- Never skip a task
- Never proceed to the next task if QA fails
- All code must match the file structure in the PRD
- No external LLM APIs, no LangChain, no LlamaIndex
- Use only `requests` for HTTP — no `httpx`, no `aiohttp`
- Every file must have a `if __name__ == "__main__"` smoke test block

---

## Project Structure (Create First)

```
wikipedia-rag/
├── README.md                 (written last)
├── product_prd.md            (already exists)
├── recommendation.md         (written last)
├── requirements.txt
├── TASKS.md                  (this file)
│
├── data/                     (create empty dir)
├── chroma_store/             (create empty dir)
│
├── src/
│   ├── __init__.py
│   ├── entities.py
│   ├── database.py
│   ├── fetcher.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
│
├── ingest.py
└── app.py
```

---

## TASK 0 — Project Scaffold

### Builder Instructions
1. Create all directories and empty files listed in the project structure above
2. Create `requirements.txt` with the following contents:
```
chromadb
streamlit
requests
```
3. Create `src/__init__.py` as an empty file
4. Create `src/entities.py` with two lists:

```python
PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
    "William Shakespeare", "Ada Lovelace", "Nikola Tesla",
    "Lionel Messi", "Cristiano Ronaldo", "Taylor Swift",
    "Frida Kahlo", "Isaac Newton", "Charles Darwin",
    "Cleopatra", "Napoleon Bonaparte", "Mahatma Gandhi",
    "Nelson Mandela", "Aristotle", "Julius Caesar",
    "Wolfgang Amadeus Mozart", "Galileo Galilei"
]

PLACES = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal",
    "Grand Canyon", "Machu Picchu", "Colosseum",
    "Hagia Sophia", "Statue of Liberty", "Pyramids of Giza",
    "Mount Everest", "Stonehenge", "Angkor Wat",
    "Acropolis of Athens", "Chichen Itza", "Petra",
    "Sydney Opera House", "Venice", "Santorini",
    "Niagara Falls", "Amazon River"
]
```

### QA Checklist — TASK 0
- [ ] All directories exist: `data/`, `chroma_store/`, `src/`
- [ ] All files exist (even if empty)
- [ ] `requirements.txt` has exactly 3 dependencies
- [ ] `src/entities.py` has exactly 20 people and 20 places
- [ ] `python -c "from src.entities import PEOPLE, PLACES; print(len(PEOPLE), len(PLACES))"` prints `20 20`

**QA PASS criteria:** All 5 checks green → proceed to Task 1

---

## TASK 1 — SQLite Database Layer

### Builder Instructions
Implement `src/database.py` with the following:

**Schema (create on first connect):**
```sql
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL CHECK(type IN ('person', 'place')),
    url         TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    token_count INTEGER,
    chroma_id   TEXT NOT NULL UNIQUE
);
```

**Functions to implement:**
```python
def get_connection(db_path: str = "data/wikipedia.db") -> sqlite3.Connection
def init_db(db_path: str = "data/wikipedia.db") -> None
def entity_exists(conn, title: str) -> bool
def insert_entity(conn, title: str, type: str, url: str, raw_text: str) -> int
def insert_chunk(conn, entity_id: int, chunk_index: int, chunk_text: str, token_count: int, chroma_id: str) -> int
def get_chunk_by_chroma_id(conn, chroma_id: str) -> dict | None
def get_all_entities(conn) -> list[dict]
def get_chunks_by_entity(conn, entity_id: int) -> list[dict]
```

**Requirements:**
- Use `sqlite3` from stdlib only
- `get_connection()` must set `row_factory = sqlite3.Row` so rows behave like dicts
- `insert_entity` and `insert_chunk` must use parameterized queries (no string formatting)
- `get_chunk_by_chroma_id` returns `None` if not found, never raises
- All functions accept an explicit `conn` parameter (no global state)

**Smoke test block:**
```python
if __name__ == "__main__":
    conn = get_connection()
    init_db()
    eid = insert_entity(conn, "Test Person", "person", "http://test.com", "Some raw text here")
    print(f"Inserted entity id: {eid}")
    print(f"Exists check: {entity_exists(conn, 'Test Person')}")
    cid = insert_chunk(conn, eid, 0, "chunk text here", 3, "chroma-test-001")
    chunk = get_chunk_by_chroma_id(conn, "chroma-test-001")
    print(f"Retrieved chunk: {chunk['chunk_text']}")
    conn.close()
    print("database.py smoke test PASSED")
```

### QA Checklist — TASK 1
- [ ] `python src/database.py` runs without errors and prints `PASSED`
- [ ] `data/wikipedia.db` file exists after running
- [ ] Both tables exist: verify with `sqlite3 data/wikipedia.db ".tables"`
- [ ] `entity_exists()` returns `False` for unknown title
- [ ] `get_chunk_by_chroma_id()` returns `None` for unknown id (not an exception)
- [ ] No raw string formatting in SQL queries (check for f-strings or % in SQL strings)

**QA PASS criteria:** All 6 checks green → proceed to Task 2

---

## TASK 2 — Wikipedia Fetcher

### Builder Instructions
Implement `src/fetcher.py` using `requests` only (no `wikipedia` library, no BeautifulSoup).

**Use the Wikipedia API:**
```
https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={title}&format=json&redirects=1
```

**Functions to implement:**
```python
def fetch_wikipedia_text(title: str) -> dict:
    """
    Returns:
    {
        "title": str,        # normalized title from Wikipedia
        "url": str,          # full Wikipedia URL
        "text": str,         # full plain text of article
        "success": bool
    }
    Returns {"success": False, "title": title, "error": str} on failure.
    """

def clean_text(text: str) -> str:
    """
    Remove excessive whitespace, == Section == headers markers,
    and lines shorter than 20 characters (usually navigation artifacts).
    Return clean plain text.
    """
```

**Requirements:**
- Handle HTTP errors gracefully (try/except, return `success: False`)
- Handle missing/disambiguation pages gracefully
- Add `time.sleep(0.3)` after each request (rate limiting)
- URL must be constructed as `https://en.wikipedia.org/wiki/{title.replace(' ', '_')}`
- `clean_text` must be called inside `fetch_wikipedia_text` before returning

**Smoke test block:**
```python
if __name__ == "__main__":
    result = fetch_wikipedia_text("Albert Einstein")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"First 200 chars: {result['text'][:200]}")
    assert result["success"] == True
    assert len(result["text"]) > 1000

    fail = fetch_wikipedia_text("XYZXYZXYZ_nonexistent_page_123")
    assert fail["success"] == False
    print("fetcher.py smoke test PASSED")
```

### QA Checklist — TASK 2
- [ ] `python src/fetcher.py` runs and prints `PASSED`
- [ ] Returned text for Einstein is > 1000 characters
- [ ] Nonexistent page returns `{"success": False}` without crashing
- [ ] No `import wikipedia` or `from bs4` anywhere in the file
- [ ] `requests` is the only HTTP library imported
- [ ] `clean_text` removes lines shorter than 20 chars

**QA PASS criteria:** All 6 checks green → proceed to Task 3

---

## TASK 3 — Chunker

### Builder Instructions
Implement `src/chunker.py` with paragraph-aware chunking and overlap.

**Functions to implement:**
```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: split on whitespace, return word count. Good enough."""

def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """
    1. Split text on double newlines to get paragraphs
    2. Filter out empty paragraphs
    3. Greedily merge paragraphs into chunks up to max_tokens
    4. When a chunk would exceed max_tokens, save it and start a new one
       carrying over the last overlap_tokens worth of words from the previous chunk
    5. Return list of chunk strings
    """

def chunk_entity(title: str, text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[dict]:
    """
    Returns list of:
    {
        "chunk_index": int,
        "chunk_text": str,
        "token_count": int
    }
    """
```

**Requirements:**
- No chunking libraries (no `langchain`, no `nltk`)
- Overlap must be word-level, not character-level
- Empty or whitespace-only chunks must be filtered out
- Minimum chunk size: 20 tokens (skip smaller chunks)
- Each chunk must contain meaningful text, not just headers

**Smoke test block:**
```python
if __name__ == "__main__":
    sample = "\n\n".join([
        "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest scientists of all time.",
        "He developed the theory of relativity, one of the two pillars of modern physics, alongside quantum mechanics.",
        "His work is also known for its influence on the philosophy of science.",
        "He received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
    ] * 10)  # repeat to get enough text

    chunks = chunk_text(sample, max_tokens=100, overlap_tokens=20)
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {estimate_tokens(c)} tokens")
    
    assert len(chunks) > 1, "Should produce multiple chunks"
    assert all(estimate_tokens(c) <= 120 for c in chunks), "No chunk should far exceed max_tokens"
    print("chunker.py smoke test PASSED")
```

### QA Checklist — TASK 3
- [ ] `python src/chunker.py` prints `PASSED`
- [ ] Multiple chunks produced from long text
- [ ] No chunk significantly exceeds `max_tokens` (allow 20% tolerance)
- [ ] No empty chunks in output
- [ ] Overlap logic present in code (last N words carried to next chunk)
- [ ] `chunk_entity` returns dicts with `chunk_index`, `chunk_text`, `token_count` keys

**QA PASS criteria:** All 6 checks green → proceed to Task 4

---

## TASK 4 — Embedder

### Builder Instructions
Implement `src/embedder.py` using Ollama's HTTP API directly.

**Ollama embedding endpoint:**
```
POST http://localhost:11434/api/embeddings
Body: {"model": "nomic-embed-text", "prompt": "<text>"}
Response: {"embedding": [float, float, ...]}  # 768 dimensions
```

**Functions to implement:**
```python
def check_ollama_running() -> bool:
    """
    GET http://localhost:11434/
    Returns True if Ollama is reachable, False otherwise.
    Never raises.
    """

def embed_text(text: str, model: str = "nomic-embed-text") -> list[float] | None:
    """
    Returns 768-dim embedding list, or None on failure.
    """

def embed_batch(texts: list[str], model: str = "nomic-embed-text") -> list[list[float] | None]:
    """
    Embed multiple texts. Add time.sleep(0.1) between calls.
    Returns list of embeddings (None for any that failed).
    """
```

**Requirements:**
- Use `requests` only
- `check_ollama_running` must be called at module import time and print a warning if Ollama is not running
- Never raise an exception to the caller — return `None` on any failure
- Log failed embeddings with `print(f"Warning: embedding failed for text starting with: {text[:50]}")`

**Smoke test block:**
```python
if __name__ == "__main__":
    if not check_ollama_running():
        print("ERROR: Ollama is not running. Start with: ollama serve")
        exit(1)
    
    emb = embed_text("Albert Einstein was a physicist")
    print(f"Embedding dimension: {len(emb)}")
    assert len(emb) == 768
    assert all(isinstance(x, float) for x in emb)

    batch = embed_batch(["Eiffel Tower is in Paris", "Marie Curie discovered radium"])
    print(f"Batch size: {len(batch)}")
    assert len(batch) == 2
    assert all(len(e) == 768 for e in batch)
    print("embedder.py smoke test PASSED")
```

### QA Checklist — TASK 4
- [ ] `python src/embedder.py` prints `PASSED` (requires Ollama running)
- [ ] Embedding dimension is exactly 768
- [ ] `check_ollama_running()` returns `False` gracefully when Ollama is off
- [ ] `embed_text` returns `None` on failure, does not raise
- [ ] No hardcoded port other than 11434
- [ ] `embed_batch` has sleep between calls

**QA PASS criteria:** All 6 checks green → proceed to Task 5

---

## TASK 5 — Vector Store

### Builder Instructions
Implement `src/vector_store.py` wrapping ChromaDB.

**Functions to implement:**
```python
def get_collection(persist_dir: str = "chroma_store") -> chromadb.Collection:
    """
    Initialize ChromaDB persistent client.
    Get or create collection named "wikipedia_rag".
    Return the collection object.
    """

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

def collection_count(collection) -> int:
    """Return total number of documents in collection."""

def chunk_already_indexed(collection, chroma_id: str) -> bool:
    """Return True if chroma_id already exists in collection."""
```

**Requirements:**
- Use `chromadb` library
- Persist to disk (not in-memory)
- `query_collection` must handle the case where `n_results` > collection size gracefully
- `chunk_already_indexed` prevents duplicate embeddings on re-run

**Smoke test block:**
```python
if __name__ == "__main__":
    import shutil
    # Use a temp test directory
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
```

### QA Checklist — TASK 5
- [ ] `python src/vector_store.py` prints `PASSED`
- [ ] ChromaDB persists to disk (not in-memory)
- [ ] `filter_type` filtering works (only persons returned when `filter_type="person"`)
- [ ] Duplicate `chroma_id` does not crash — `chunk_already_indexed` prevents it
- [ ] `query_collection` returns correct dict structure
- [ ] Test directory is cleaned up after smoke test

**QA PASS criteria:** All 6 checks green → proceed to Task 6

---

## TASK 6 — Ingestion Pipeline

### Builder Instructions
Implement `ingest.py` as the main ingestion script. This orchestrates all previous modules.

**Pipeline per entity:**
```
1. Check SQLite: if entity already ingested, skip
2. Fetch Wikipedia text (fetcher.py)
3. Store entity in SQLite (database.py)
4. Chunk the text (chunker.py)
5. For each chunk:
   a. Generate chroma_id = f"{title.replace(' ', '_')}_{chunk_index}"
   b. Check if chroma_id already in ChromaDB (skip if yes)
   c. Embed chunk text (embedder.py)
   d. Store chunk in SQLite (database.py)
   e. Store embedding in ChromaDB (vector_store.py)
6. Print progress
```

**Script behavior:**
```python
# Run with: python ingest.py
# Should be safe to run multiple times (idempotent)
# Print progress like:
# [1/40] Ingesting Albert Einstein (person)... 14 chunks embedded. Done.
# [2/40] Albert Einstein already ingested. Skipping.
```

**Requirements:**
- Must be idempotent — safe to run multiple times
- Handle individual entity failures gracefully (log and continue)
- Print final summary: `Ingestion complete. X entities, Y chunks total.`
- Commit SQLite transaction per entity (not per chunk)
- Check Ollama is running before starting; exit with clear error if not

**No smoke test needed** — the full run IS the test.

### QA Checklist — TASK 6
- [ ] `python ingest.py` runs without crashing for at least 5 test entities
- [ ] Running `python ingest.py` a second time skips already-ingested entities
- [ ] `sqlite3 data/wikipedia.db "SELECT count(*) FROM entities"` returns expected count
- [ ] `sqlite3 data/wikipedia.db "SELECT count(*) FROM chunks"` returns > 0
- [ ] ChromaDB collection count matches SQLite chunks count
- [ ] Failed entity (if any) is logged but doesn't stop ingestion

**QA PASS criteria:** All 6 checks green → proceed to Task 7

---

## TASK 7 — Retriever (Query Classifier + Retrieval)

### Builder Instructions
Implement `src/retriever.py`.

**Query classification logic:**

```python
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
```

**Functions to implement:**
```python
def classify_query(query: str, people: list[str], places: list[str]) -> str:
    """
    Returns "person", "place", or "both".
    
    Logic (in order):
    1. Check if any person name appears in query → lean toward "person"
    2. Check if any place name appears in query → lean toward "place"
    3. If both names found → "both"
    4. Count keyword matches for person vs place keywords
    5. If person_score > place_score → "person"
    6. If place_score > person_score → "place"
    7. Default → "both"
    """

def retrieve(
    query: str,
    collection,
    conn,
    n_results: int = 5
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Classify query
    2. Embed query using embedder.embed_text()
    3. Query ChromaDB with optional type filter
    4. For each result, fetch chunk text from SQLite
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
```

**Smoke test block:**
```python
if __name__ == "__main__":
    from src.entities import PEOPLE, PLACES

    # Test classifier only (no DB needed)
    assert classify_query("Who was Albert Einstein?", PEOPLE, PLACES) == "person"
    assert classify_query("Where is the Eiffel Tower?", PEOPLE, PLACES) == "place"
    assert classify_query("Compare Einstein and the Eiffel Tower", PEOPLE, PLACES) == "both"
    assert classify_query("something completely unrelated xyz", PEOPLE, PLACES) == "both"
    print("Classifier tests PASSED")
    print("retriever.py smoke test PASSED")
```

### QA Checklist — TASK 7
- [ ] `python src/retriever.py` prints `PASSED`
- [ ] "Who was Einstein" → classified as "person"
- [ ] "Where is Eiffel Tower" → classified as "place"
- [ ] "Compare Einstein and Tesla" → classified as "person" or "both" (not "place")
- [ ] "Who is the president of Mars" → classified as "both" (unknown → default)
- [ ] `retrieve()` returns dicts with all required keys

**QA PASS criteria:** All 6 checks green → proceed to Task 8

---

## TASK 8 — Generator

### Builder Instructions
Implement `src/generator.py`.

**Prompt template:**
```
You are a factual assistant. Answer the question using ONLY the context below.
If the answer cannot be found in the context, respond with exactly:
"I don't know based on the available information."
Do not add information from outside the context. Be concise.

Context:
{context}

Question: {query}

Answer:
```

**Functions to implement:**
```python
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Format the prompt template with query and up to 5 chunks.
    Join chunks with "---" separator.
    Include entity_title as a header before each chunk:
    "[Albert Einstein]\nchunk text here"
    """

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str = "tinyllama",
    max_tokens: int = 250
) -> dict:
    """
    Call Ollama /api/generate endpoint.
    
    Returns:
    {
        "answer": str,
        "prompt_used": str,
        "model": str,
        "chunks_used": int
    }
    
    If no chunks retrieved, return "I don't know" without calling Ollama.
    If Ollama fails, return error message as answer.
    """
```

**Ollama generate endpoint:**
```
POST http://localhost:11434/api/generate
Body: {
    "model": "tinyllama",
    "prompt": "...",
    "stream": false,
    "options": {"num_predict": 250, "temperature": 0.1}
}
Response: {"response": "...", ...}
```

**Requirements:**
- `temperature: 0.1` for factual consistency
- `stream: false` for simplicity
- If `retrieved_chunks` is empty → return "I don't know" immediately
- Strip `<|im_end|>` and similar special tokens from tinyllama output

**Smoke test block:**
```python
if __name__ == "__main__":
    # Test prompt building without Ollama
    fake_chunks = [
        {"entity_title": "Albert Einstein", "chunk_text": "Einstein developed the theory of relativity.", "distance": 0.1},
        {"entity_title": "Albert Einstein", "chunk_text": "He was born in Ulm, Germany in 1879.", "distance": 0.2}
    ]
    prompt = build_prompt("Who was Albert Einstein?", fake_chunks)
    print("Prompt preview:")
    print(prompt[:300])
    assert "theory of relativity" in prompt
    assert "Albert Einstein" in prompt

    # Test empty chunks fallback
    result = generate_answer("test query", [], model="tinyllama")
    assert "don't know" in result["answer"].lower()
    print("generator.py smoke test PASSED")
```

### QA Checklist — TASK 8
- [ ] `python src/generator.py` prints `PASSED`
- [ ] Empty chunks → returns "I don't know" without calling Ollama
- [ ] Prompt contains entity title headers
- [ ] `temperature` is 0.1 in the API call
- [ ] `stream` is `false` in the API call
- [ ] Special tokens stripped from output

**QA PASS criteria:** All 6 checks green → proceed to Task 9

---

## TASK 9 — Streamlit Chat Application

### Builder Instructions
Implement `app.py` as the main Streamlit chat interface.

**UI Layout:**
```
┌─────────────────────────────────────────┐
│  🧠 Wikipedia RAG  [Clear Chat] button  │
├─────────────────────────────────────────┤
│  Sidebar:                               │
│  - Ollama status (✅ / ❌)              │
│  - DB stats (entities, chunks)          │
│  - Show retrieved context toggle        │
│  - Model info (tinyllama)               │
├─────────────────────────────────────────┤
│  Chat history (user + assistant msgs)   │
├─────────────────────────────────────────┤
│  [Text input box]                       │
└─────────────────────────────────────────┘
```

**Requirements:**
- Use `st.session_state` for chat history
- Use `st.chat_message` for rendering messages
- Show spinner with "Thinking..." while generating
- If "Show retrieved context" is toggled on, show expandable section below each answer with source chunks
- "Clear Chat" button resets only chat history (not the DB)
- On startup, check Ollama status and show warning if offline
- Handle empty query gracefully (don't submit)
- Show query classification result (e.g. "🔍 Searching: person") in the spinner

**Session state keys:**
```python
st.session_state.messages       # list of {"role": "user"|"assistant", "content": str, "chunks": list}
st.session_state.show_context   # bool
```

**No smoke test for Streamlit** — run with `streamlit run app.py` to verify.

### QA Checklist — TASK 9
- [ ] `streamlit run app.py` starts without import errors
- [ ] Chat history persists across multiple questions in session
- [ ] "Clear Chat" clears messages without page reload issues
- [ ] Sidebar shows correct entity/chunk counts from SQLite
- [ ] Retrieved context section appears when toggle is on
- [ ] Empty query does not trigger generation
- [ ] Ollama offline warning shown in sidebar

**QA PASS criteria:** All 7 checks green → proceed to Integration Test

---

## FINAL — Integration Test

Run the full system end-to-end. QA agent verifies all of the following:

### Setup
```bash
ollama serve &
python ingest.py
streamlit run app.py
```

### Test Queries (run each, verify answer quality)

| Query | Expected behavior |
|-------|------------------|
| "Who was Albert Einstein?" | Returns biography info, source = person |
| "Where is the Eiffel Tower?" | Returns location info, source = place |
| "What did Marie Curie discover?" | Returns radium/polonium info |
| "What was the Colosseum used for?" | Returns gladiatorial contests info |
| "Compare Messi and Ronaldo" | Returns info about both |
| "Which famous place is in Turkey?" | Returns Hagia Sophia |
| "Who is the president of Mars?" | Returns "I don't know" |
| "Tell me about John Doe" | Returns "I don't know" |

### Integration QA Checklist
- [ ] All 8 test queries return without crashing
- [ ] "President of Mars" returns "I don't know"
- [ ] "John Doe" returns "I don't know"
- [ ] Eiffel Tower query uses `filter_type="place"`
- [ ] Einstein query uses `filter_type="person"`
- [ ] Re-running `python ingest.py` skips all entities (idempotent)
- [ ] `streamlit run app.py` works from a fresh terminal after ingestion

**ALL GREEN = Project complete. Proceed to README.md and recommendation.md.**

---

## README.md Outline (write last)

```markdown
# Wikipedia RAG — Local AI Question Answering

## Prerequisites
- Python 3.10+
- Ollama installed (https://ollama.com/install)

## Setup
1. Clone repo
2. pip install -r requirements.txt
3. ollama pull tinyllama
4. ollama pull nomic-embed-text
5. python ingest.py
6. streamlit run app.py

## Example Questions
...

## Architecture
...
```

## recommendation.md Outline (write last)

Cover:
- Replace tinyllama with llama3 or mistral for better quality
- Replace nomic-embed-text with OpenAI ada-002 for better retrieval
- Move ChromaDB to Pinecone/Weaviate for scale
- Add Redis cache for repeated queries
- Containerize with Docker
- Add authentication for multi-user
- Latency tradeoffs observed during development