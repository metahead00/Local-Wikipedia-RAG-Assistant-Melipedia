"""
embedder.py — Ollama-backed text embedder for the Wikipedia RAG pipeline.

Uses the Ollama HTTP API directly via `requests`.
Default model: nomic-embed-text (768-dim embeddings).
"""

import time
import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_running() -> bool:
    """
    GET http://localhost:11434/
    Returns True if Ollama is reachable, False otherwise.
    Never raises.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def embed_text(
    text: str,
    model: str = "nomic-embed-text",
    task: str = "",
) -> list[float] | None:
    """
    Embed a single text string using Ollama.
    Returns a 768-dim embedding list, or None on failure.
    Never raises an exception to the caller.
    """
    prompt = f"{task}: {text}" if task else text
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": prompt},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except Exception:
        print(f"Warning: embedding failed for text starting with: {text[:50]}")
        return None


def embed_batch(
    texts: list[str],
    model: str = "nomic-embed-text",
    task: str = "",
) -> list[list[float] | None]:
    """
    Embed multiple texts sequentially with a small sleep between calls.

    Returns a list of embeddings (None for any that failed).
    Never raises an exception to the caller.
    """
    results: list[list[float] | None] = []
    for i, text in enumerate(texts):
        embedding = embed_text(text, model=model, task=task)
        results.append(embedding)
        if i < len(texts) - 1:
            time.sleep(0.1)
    return results


# Module-level check: warn if Ollama is not reachable at import time.
if not check_ollama_running():
    print(
        "Warning: Ollama does not appear to be running at "
        f"{OLLAMA_BASE_URL}. Embeddings will fail until Ollama is started."
    )


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
