from .models import Document


def chunk_document(document: Document, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = document.text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
