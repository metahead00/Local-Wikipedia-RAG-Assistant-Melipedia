"""
database.py — SQLite database layer for the Wikipedia RAG project.

Provides connection management, schema initialisation, and CRUD helpers
for the `entities` and `chunks` tables.
"""

import sqlite3


# ---------------------------------------------------------------------------
# Connection & schema
# ---------------------------------------------------------------------------

def get_connection(db_path: str = "data/wikipedia.db") -> sqlite3.Connection:
    """Return an open sqlite3 connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "data/wikipedia.db") -> None:
    """Create both tables (entities and chunks) if they do not yet exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(
            """
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
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------

def entity_exists(conn: sqlite3.Connection, title: str) -> bool:
    """Return True if an entity with the given title already exists."""
    cursor = conn.execute(
        "SELECT 1 FROM entities WHERE title = ?",
        (title,),
    )
    return cursor.fetchone() is not None


def insert_entity(
    conn: sqlite3.Connection,
    title: str,
    type: str,
    url: str,
    raw_text: str,
) -> int:
    """Insert a new entity row and return its auto-assigned id."""
    cursor = conn.execute(
        "INSERT INTO entities (title, type, url, raw_text) VALUES (?, ?, ?, ?)",
        (title, type, url, raw_text),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def insert_chunk(
    conn: sqlite3.Connection,
    entity_id: int,
    chunk_index: int,
    chunk_text: str,
    token_count: int,
    chroma_id: str,
) -> int:
    """Insert a new chunk row and return its auto-assigned id."""
    cursor = conn.execute(
        "INSERT INTO chunks (entity_id, chunk_index, chunk_text, token_count, chroma_id)"
        " VALUES (?, ?, ?, ?, ?)",
        (entity_id, chunk_index, chunk_text, token_count, chroma_id),
    )
    conn.commit()
    return cursor.lastrowid


def get_chunk_by_chroma_id(
    conn: sqlite3.Connection, chroma_id: str
) -> dict | None:
    """Return the chunk row matching chroma_id as a dict, or None if not found."""
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE chroma_id = ?",
        (chroma_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def get_all_entities(conn: sqlite3.Connection) -> list[dict]:
    """Return all entity rows as a list of dicts."""
    cursor = conn.execute("SELECT * FROM entities")
    return [dict(row) for row in cursor.fetchall()]


def get_chunks_by_entity(conn: sqlite3.Connection, entity_id: int) -> list[dict]:
    """Return all chunk rows for a given entity_id as a list of dicts."""
    cursor = conn.execute(
        "SELECT * FROM chunks WHERE entity_id = ?",
        (entity_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

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
