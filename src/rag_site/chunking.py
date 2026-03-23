def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])

        if end >= n:
            break

        start = end - overlap

    return chunks
