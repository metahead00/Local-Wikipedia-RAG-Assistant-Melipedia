"""
chunker.py — Paragraph-aware text chunking with overlap.
No external chunking libraries required.
"""


def estimate_tokens(text: str) -> int:
    """Rough estimate: split on whitespace, return word count."""
    return len(text.split())


def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    """
    1. Split text on double newlines to get paragraphs
    2. Filter out empty paragraphs
    3. Greedily merge paragraphs into chunks up to max_tokens
    4. When a chunk would exceed max_tokens, save it and start a new one
       carrying over the last overlap_tokens worth of words from the previous chunk
    5. Return list of chunk strings
    """
    # Step 1 & 2: Split on double newlines and filter empty paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chunks = []
    current_words: list[str] = []

    for paragraph in paragraphs:
        para_words = paragraph.split()

        # If a single paragraph exceeds max_tokens on its own, we need to handle it
        # by splitting it word-by-word as well
        if not current_words and estimate_tokens(paragraph) > max_tokens:
            # Flush the oversized paragraph in segments
            i = 0
            while i < len(para_words):
                segment = para_words[i : i + max_tokens]
                chunk_str = " ".join(segment)
                if estimate_tokens(chunk_str) >= 20:
                    chunks.append(chunk_str)
                # Advance by max_tokens minus overlap so next segment has overlap
                advance = max(1, max_tokens - overlap_tokens)
                i += advance
            # After flushing, carry over overlap from the last segment
            if chunks:
                last_chunk_words = chunks[-1].split()
                current_words = last_chunk_words[-overlap_tokens:] if overlap_tokens else []
            continue

        # Would adding this paragraph exceed max_tokens?
        if current_words and (len(current_words) + len(para_words)) > max_tokens:
            # Save current chunk
            chunk_str = " ".join(current_words)
            if estimate_tokens(chunk_str) >= 20:
                chunks.append(chunk_str)

            # Start new chunk with overlap from the end of current chunk
            overlap_words = current_words[-overlap_tokens:] if overlap_tokens else []
            current_words = overlap_words + para_words
        else:
            # Merge paragraph into current chunk
            current_words.extend(para_words)

        # If current_words already exceeds max_tokens after merging, flush it
        while len(current_words) > max_tokens:
            segment_words = current_words[:max_tokens]
            chunk_str = " ".join(segment_words)
            if estimate_tokens(chunk_str) >= 20:
                chunks.append(chunk_str)
            advance = max(1, max_tokens - overlap_tokens)
            current_words = current_words[advance:]

    # Flush remaining words
    if current_words:
        chunk_str = " ".join(current_words)
        if estimate_tokens(chunk_str) >= 20:
            chunks.append(chunk_str)

    return chunks


def chunk_entity(title: str, text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[dict]:
    """
    Returns list of:
    {
        "chunk_index": int,
        "chunk_text": str,
        "token_count": int
    }
    """
    raw_chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    result = []
    for idx, chunk_str in enumerate(raw_chunks):
        result.append(
            {
                "chunk_index": idx,
                "chunk_text": chunk_str,
                "token_count": estimate_tokens(chunk_str),
            }
        )
    return result


if __name__ == "__main__":
    sample = "\n\n".join(
        [
            "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest scientists of all time.",
            "He developed the theory of relativity, one of the two pillars of modern physics, alongside quantum mechanics.",
            "His work is also known for its influence on the philosophy of science.",
            "He received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        ]
        * 10
    )  # repeat to get enough text

    chunks = chunk_text(sample, max_tokens=100, overlap_tokens=20)
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {estimate_tokens(c)} tokens")

    assert len(chunks) > 1, "Should produce multiple chunks"
    assert all(estimate_tokens(c) <= 120 for c in chunks), "No chunk should far exceed max_tokens"
    print("chunker.py smoke test PASSED")
