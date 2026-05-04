"""
Answer generator module for Wikipedia RAG.
Builds prompts from retrieved chunks and calls Ollama to generate answers.
"""

import requests

SYSTEM_PROMPT = (
    "You are a factual assistant. Answer the user's question using ONLY the provided context. "
    "Write 2-4 complete sentences. "
    "If the answer is not in the context, respond with exactly: "
    "'I don't know based on the available information.' "
    "Do not use outside knowledge."
)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GENERATION_MODEL = "llama3.2"


def build_user_message(query: str, retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks + query into the user turn of the chat."""
    chunks = retrieved_chunks[:5]
    chunk_parts = []
    for chunk in chunks:
        title = chunk.get("entity_title", "Unknown")
        text = chunk.get("chunk_text", "")
        chunk_parts.append(f"[{title}]\n{text}")

    context = "\n---\n".join(chunk_parts)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str = GENERATION_MODEL,
    max_tokens: int = 300,
) -> dict:
    """
    Call Ollama /api/chat endpoint with system + user messages.

    Returns:
    {
        "answer": str,
        "prompt_used": str,
        "model": str,
        "chunks_used": int
    }

    If no chunks retrieved, return "I don't know based on the available information."
    without calling Ollama.
    If Ollama fails, return error message as answer.
    """
    if not retrieved_chunks:
        return {
            "answer": "I don't know based on the available information.",
            "prompt_used": "",
            "model": model,
            "chunks_used": 0,
        }

    user_message = build_user_message(query, retrieved_chunks)
    chunks_used = min(len(retrieved_chunks), 5)

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as exc:
        answer = f"Error contacting Ollama: {exc}"

    return {
        "answer": answer,
        "prompt_used": user_message,
        "model": model,
        "chunks_used": chunks_used,
    }


if __name__ == "__main__":
    fake_chunks = [
        {
            "entity_title": "Albert Einstein",
            "chunk_text": "Einstein developed the theory of relativity.",
            "distance": 0.1,
        },
        {
            "entity_title": "Albert Einstein",
            "chunk_text": "He was born in Ulm, Germany in 1879.",
            "distance": 0.2,
        },
    ]
    msg = build_user_message("Who was Albert Einstein?", fake_chunks)
    print("User message preview:")
    print(msg[:300])
    assert "theory of relativity" in msg
    assert "Albert Einstein" in msg

    result = generate_answer("test query", [])
    assert "don't know" in result["answer"].lower()
    print("generator.py smoke test PASSED")
