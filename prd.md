# Product Requirements Document
## Local Wikipedia RAG System
**Course:** BLG 483E — AI-Aided Computer Engineering  
**Assignment:** Project 3  
**Author:** Melike  
**Version:** 1.0

---

## 1. Overview

This document defines the requirements for a locally-running Retrieval-Augmented Generation (RAG) system that answers natural language questions about famous people and places using Wikipedia as its knowledge source. The system runs entirely on localhost with no external API dependencies.

The system is a simplified, transparent implementation of how production AI assistants like ChatGPT work under the hood: ingest → chunk → embed → retrieve → generate.

---

## 2. Goals

- Build a fully offline, CPU-runnable question-answering system
- Demonstrate understanding of the RAG pipeline end-to-end
- Reuse and extend the Wikipedia fetching logic from Project 1
- Avoid black-box libraries — implement core logic (chunking, retrieval routing, query classification) from scratch
- Produce a clean, demo-ready Streamlit UI

---

## 3. Non-Goals

- No external LLM APIs (OpenAI, Anthropic, etc.)
- No GPU requirement
- No LangChain, LlamaIndex, or similar RAG frameworks doing the core work
- No real-time Wikipedia updates during chat

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                  │
│                                                         │
│  Wikipedia API  →  Fetcher  →  Chunker  →  Embedder    │
│  (requests)        (HW1        (custom     (Ollama      │
│                    adapted)    logic)       nomic)      │
│                         ↓              ↓                │
│                      SQLite        ChromaDB             │
│                   (raw text,     (embeddings,           │
│                    metadata)      chunk IDs)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                     │
│                                                         │
│  User Query → Classifier → ChromaDB query              │
│                 (rule-        (filtered by              │
│                 based)         metadata)                │
│                                    ↓                    │
│                             SQLite lookup               │
│                          (fetch chunk text)             │
│                                    ↓                    │
│                          Ollama tinyllama               │
│                          (grounded answer)              │
│                                    ↓                    │
│                          Streamlit UI                   │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack

| Component        | Choice                        | Reason |
|-----------------|-------------------------------|--------|
| Language        | Python 3.10+                  | Required |
| LLM             | tinyllama via Ollama          | Fastest CPU inference (~10s/response), no GPU needed |
| Embeddings      | nomic-embed-text via Ollama   | 768-dim, local, no API, consistent stack |
| Vector Store    | ChromaDB                      | Required by assignment |
| Relational DB   | SQLite                        | Lightweight, no server, stores raw text and metadata |
| UI              | Streamlit                     | Fast to build, clean chat interface |
| HTTP Client     | requests (stdlib)             | Consistent with HW1, no extra dependencies |
| Wikipedia Source| Wikipedia REST API            | Clean JSON, no scraping needed |

---

## 6. Design Decisions

### 6.1 Single Vector Store with Metadata (Option B)

**Decision:** One ChromaDB collection with a `type` metadata field (`"person"` or `"place"`).

**Rationale:**
- Architecturally simpler — one collection to maintain, one embedding space
- Metadata filtering in ChromaDB is a first-class feature (`where={"type": "person"}`)
- Scales better: adding a third category (e.g. "event") requires no structural change
- Mirrors how production systems work (e.g. filtering by tenant, category, date)
- Avoids duplicating collection management logic

**Tradeoff:** Slightly more complex query routing vs Option A's naive "just query the right store."

---

### 6.2 Chunking Strategy: Paragraph-Aware with Token Overlap

**Decision:** Split Wikipedia text on double newlines (natural paragraph boundaries), then enforce a max chunk size of ~400 tokens with 50-token overlap between consecutive chunks.

**Rationale:**
- Wikipedia articles use paragraphs as natural semantic units — splitting on `\n\n` preserves meaning better than fixed character counts
- Max size cap prevents oversized chunks overwhelming the LLM context window
- Overlap ensures facts that span paragraph boundaries aren't lost
- Implemented from scratch using basic string operations — no chunking library

**Chunk size justification:** tinyllama has a 2048-token context window. With ~5 chunks retrieved, we need each chunk under ~300-400 tokens to leave room for the prompt template and generated answer.

---

### 6.3 Wikipedia Fetching via REST API

**Decision:** Use `https://en.wikipedia.org/w/api.php` with `action=query&prop=extracts` to fetch full article text, adapted from HW1's generic crawler.

**Rationale:**
- Consistent with HW1 philosophy: `requests` only, no external library
- Wikipedia's API returns clean, structured text — no HTML parsing needed
- `explaintext=1` parameter strips markup automatically

---

### 6.4 Query Classification: Rule-Based

**Decision:** Classify user queries as `person`, `place`, or `both` using:
1. A hardcoded name list (all ingested entities)
2. Keyword triggers (`"where is"`, `"located"`, `"city"`, `"country"` → place; `"who was"`, `"born"`, `"discovered"` → person)
3. Default to `both` if ambiguous

**Rationale:**
- Assignment explicitly allows keyword/rule-based approaches
- Lightweight — no model inference for classification
- Deterministic and debuggable
- For a known, fixed entity set, name matching is highly reliable

---

### 6.5 SQLite as Source of Truth

**Decision:** Store all ingested data (raw article text, chunks, metadata) in SQLite alongside ChromaDB.

**Rationale:**
- ChromaDB metadata storage is limited and not designed for full-text retrieval
- SQLite gives full query flexibility (filter by entity name, type, ingestion date)
- Chunk text lives in SQLite; ChromaDB stores only the embedding + chunk ID
- Decouples the embedding index from the actual content — can re-embed without re-fetching Wikipedia
- Enables cache checking: skip re-ingesting already-fetched entities

---

## 7. Data Requirements

### 7.1 Minimum Entity Set

**People (20+):**
Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Isaac Newton, Charles Darwin, Cleopatra, Napoleon Bonaparte, Mahatma Gandhi, Nelson Mandela, Aristotle, Julius Caesar, Mozart, Galileo Galilei

**Places (20+):**
Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Stonehenge, Angkor Wat, Acropolis of Athens, Chichen Itza, Petra, Sydney Opera House, Venice, Santorini, Niagara Falls, Amazon Rainforest

### 7.2 SQLite Schema

```sql
-- Stores one row per ingested Wikipedia entity
CREATE TABLE entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL CHECK(type IN ('person', 'place')),
    url         TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stores one row per chunk derived from an entity
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    token_count INTEGER,
    chroma_id   TEXT NOT NULL UNIQUE  -- foreign key into ChromaDB
);
```

### 7.3 ChromaDB Schema

**Collection name:** `wikipedia_rag`

**Document:** chunk text (for ChromaDB's internal use)

**Metadata per chunk:**
```json
{
  "entity_title": "Albert Einstein",
  "type": "person",
  "chunk_index": 3,
  "sqlite_chunk_id": 47
}
```

**Embedding:** 768-dim vector from nomic-embed-text

---

## 8. Functional Requirements

### 8.1 Ingestion

| ID | Requirement |
|----|-------------|
| ING-01 | Fetch Wikipedia article text using REST API via `requests` only |
| ING-02 | Skip already-ingested entities (check SQLite before fetching) |
| ING-03 | Store raw text and metadata in SQLite `entities` table |
| ING-04 | Split text into paragraph-aware chunks with max 400 tokens and 50-token overlap |
| ING-05 | Store all chunks in SQLite `chunks` table |
| ING-06 | Generate embeddings for each chunk using nomic-embed-text via Ollama |
| ING-07 | Store embeddings in ChromaDB with full metadata |
| ING-08 | Support ingestion of additional entities beyond the minimum set |
| ING-09 | Provide a standalone ingestion script (`ingest.py`) |

### 8.2 Retrieval

| ID | Requirement |
|----|-------------|
| RET-01 | Classify query as `person`, `place`, or `both` using rule-based classifier |
| RET-02 | Filter ChromaDB query by `type` metadata based on classification |
| RET-03 | Retrieve top-5 most relevant chunks by default |
| RET-04 | Fetch full chunk text from SQLite using `sqlite_chunk_id` from ChromaDB results |
| RET-05 | Return source entity name alongside each retrieved chunk |

### 8.3 Generation

| ID | Requirement |
|----|-------------|
| GEN-01 | Construct a prompt from query + retrieved chunks |
| GEN-02 | Send prompt to tinyllama via Ollama HTTP API |
| GEN-03 | Return "I don't know based on the available information." if no relevant chunks found |
| GEN-04 | Ground answers strictly in retrieved context — prompt must instruct model not to hallucinate |
| GEN-05 | Limit generated response to ~200 tokens for speed |

### 8.4 Chat Interface

| ID | Requirement |
|----|-------------|
| UI-01 | Streamlit app with chat message history |
| UI-02 | User can type questions and receive answers |
| UI-03 | Optionally toggle "show retrieved context" to view source chunks |
| UI-04 | "Clear conversation" button resets chat history |
| UI-05 | Show entity source name next to each retrieved chunk |
| UI-06 | Display a spinner/status message while generating |

---

## 9. Prompt Template

```
You are a knowledgeable assistant. Answer the user's question using ONLY 
the context provided below. If the answer is not in the context, say 
"I don't know based on the available information." Do not make up facts.

Context:
{chunk_1}
---
{chunk_2}
---
{chunk_3}

Question: {user_query}

Answer:
```

---

## 10. Project Structure

```
wikipedia-rag/
├── README.md
├── product_prd.md
├── recommendation.md
├── requirements.txt
│
├── data/
│   └── wikipedia.db          # SQLite database (gitignored)
│
├── chroma_store/             # ChromaDB persistent storage (gitignored)
│
├── src/
│   ├── fetcher.py            # Wikipedia API fetcher (adapted from HW1)
│   ├── chunker.py            # Paragraph-aware chunking logic
│   ├── embedder.py           # nomic-embed-text via Ollama
│   ├── database.py           # SQLite operations
│   ├── vector_store.py       # ChromaDB operations
│   ├── retriever.py          # Query classifier + retrieval logic
│   ├── generator.py          # tinyllama prompt construction + generation
│   └── entities.py           # List of all entities to ingest
│
├── ingest.py                 # Standalone ingestion script
└── app.py                    # Streamlit chat application
```

---

## 11. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Ingestion time | < 5 min for full 40+ entity set |
| Embedding generation | < 2s per chunk |
| Retrieval latency | < 1s |
| Generation latency | < 15s on CPU (tinyllama) |
| Disk usage | < 500MB total (DB + ChromaDB + models excluded) |
| Reproducibility | Instructor can run from README only |

---

## 12. Out of Scope (Optional Extensions)

These are not required but may be implemented if time allows:
- Streaming responses from tinyllama
- Chat history memory (passing previous Q&A into context)
- Response caching (SQLite cache keyed on query hash)
- Latency measurement display in UI
- Comparison query support ("Compare Einstein and Tesla")
- Second model for A/B testing (e.g. tinyllama vs phi3)

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| tinyllama hallucinating despite prompt | Strict prompt template; "I don't know" fallback |
| Wikipedia API rate limiting | Add `time.sleep(0.5)` between requests; cache in SQLite |
| ChromaDB + SQLite getting out of sync | Always write SQLite first, then ChromaDB; wrap in try/except with rollback |
| Ollama not running when app starts | Check Ollama health at startup; show clear error in Streamlit |
| Long generation time frustrating demo | Set `max_tokens=200`; show spinner with elapsed time |