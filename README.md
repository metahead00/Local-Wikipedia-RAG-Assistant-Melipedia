# Wikipedia RAG — Local AI Question Answering

A fully offline Retrieval-Augmented Generation (RAG) system that answers natural language questions about famous people and places using Wikipedia as its knowledge source. Built for BLG 483E — AI-Aided Computer Engineering, Project 3.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/install) installed and running

## Setup

1. **Clone the repo and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull the required models - change tinyllama according to your wish, if you will, change generator.py at line 48**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

4. **Ingest Wikipedia articles** (fetches and embeds all 40 entities, ~5 min)
   ```bash
   python ingest.py
   ```

5. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## Example Questions

- *Who was Albert Einstein?*
- *Where is the Eiffel Tower located?*
- *What did Marie Curie discover?*
- *What was the Colosseum used for?*
- *Compare Messi and Ronaldo*
- *Which famous place is in Turkey?*

## Architecture

```
User Query
    │
    ▼
Classifier (rule-based)
    │  person / place / both
    ▼
ChromaDB query (filtered by type)
    │  top-5 chunk IDs
    ▼
SQLite lookup (fetch chunk text)
    │  retrieved chunks
    ▼
tinyllama via Ollama (grounded generation)
    │
    ▼
Streamlit UI
```

### Ingestion pipeline

```
Wikipedia REST API → Fetcher → Chunker → nomic-embed-text → ChromaDB
                                    └──────────────────────→ SQLite
```

### Key components

| File | Role |
|------|------|
| `src/fetcher.py` | Fetches Wikipedia articles via REST API (`requests` only) |
| `src/chunker.py` | Paragraph-aware chunking, max 400 tokens, 50-token overlap |
| `src/embedder.py` | 768-dim embeddings via Ollama `nomic-embed-text` |
| `src/database.py` | SQLite layer — stores raw text, chunks, metadata |
| `src/vector_store.py` | ChromaDB wrapper — stores and queries embeddings |
| `src/retriever.py` | Rule-based query classifier + full retrieval pipeline |
| `src/generator.py` | Prompt builder + `tinyllama` generation via Ollama |
| `ingest.py` | Orchestrates the full ingestion pipeline (idempotent) |
| `app.py` | Streamlit chat interface |

## Entity Coverage

**20 people:** Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Isaac Newton, Charles Darwin, Cleopatra, Napoleon Bonaparte, Mahatma Gandhi, Nelson Mandela, Aristotle, Julius Caesar, Wolfgang Amadeus Mozart, Galileo Galilei

**20 places:** Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Stonehenge, Angkor Wat, Acropolis of Athens, Chichen Itza, Petra, Sydney Opera House, Venice, Santorini, Niagara Falls, Amazon River

## Design Decisions

- **Single ChromaDB collection** with `type` metadata filtering — simpler than two separate collections and mirrors production RAG patterns
- **Paragraph-aware chunking** — splits on `\n\n` (natural Wikipedia boundaries) before enforcing token limits
- **Rule-based classifier** — lightweight, deterministic, no model inference needed for routing
- **SQLite as source of truth** — chunk text lives in SQLite; ChromaDB holds only embeddings + IDs, keeping them decoupled
- **No external LLM APIs** — fully offline, CPU-only stack (tinyllama + nomic-embed-text via Ollama)
