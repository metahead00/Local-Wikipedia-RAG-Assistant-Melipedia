"""
Answer generator module for Wikipedia RAG.
Builds prompts from retrieved chunks and calls Ollama to generate answers.
"""

import re
import requests

PROMPT_TEMPLATE = """You are a factual assistant. Answer the question using ONLY the context below.
If the answer cannot be found in the context, respond with exactly:
"I don't know based on the available information."
Do not add information from outside the context. Be concise.

Context:
{context}

Question: {query}

Answer:"""

OLLAMA_URL = "http://localhost:11434/api/generate"

# Regex pattern to strip special tokens from tinyllama and similar models
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|]*\|>")


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Format the prompt template with query and up to 5 chunks.
    Join chunks with "---" separator.
    Include entity_title as a header before each chunk:
    "[Albert Einstein]\nchunk text here"
    """
    chunks = retrieved_chunks[:5]
    chunk_parts = []
    for chunk in chunks:
        title = chunk.get("entity_title", "Unknown")
        text = chunk.get("chunk_text", "")
        chunk_parts.append(f"[{title}]\n{text}")

    context = "\n---\n".join(chunk_parts)
    return PROMPT_TEMPLATE.format(context=context, query=query)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model: str = "tinyllama",
    max_tokens: int = 250,
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

    prompt = build_prompt(query, retrieved_chunks)
    chunks_used = min(len(retrieved_chunks), 5)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        raw_answer = data.get("response", "")
        # Strip special tokens (e.g. <|im_end|>, <|im_start|>, etc.)
        answer = SPECIAL_TOKEN_PATTERN.sub("", raw_answer).strip()
    except requests.exceptions.RequestException as exc:
        answer = f"Error contacting Ollama: {exc}"

    return {
        "answer": answer,
        "prompt_used": prompt,
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
    prompt = build_prompt("Who was Albert Einstein?", fake_chunks)
    print("Prompt preview:")
    print(prompt[:300])
    assert "theory of relativity" in prompt
    assert "Albert Einstein" in prompt

    result = generate_answer("test query", [], model="tinyllama")
    assert "don't know" in result["answer"].lower()
    print("generator.py smoke test PASSED")
