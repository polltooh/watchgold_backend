import hashlib
from urllib.parse import urlparse

from rag_site.chunking import chunk_text


def sha_id(*parts: str) -> str:
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_documents_from_pages(
    pages: list[dict],
    embedder,
) -> list[dict]:
    docs: list[dict] = []

    for page in pages:
        # metadata may be a Pydantic model or a plain dict
        raw_meta = page.get("metadata") or {}
        if hasattr(raw_meta, "model_dump"):
            meta = raw_meta.model_dump()
        elif hasattr(raw_meta, "__dict__"):
            meta = raw_meta.__dict__
        else:
            meta = dict(raw_meta)

        page_url = (
            page.get("url")
            or page.get("sourceURL")
            or page.get("source_url")
            or meta.get("sourceURL")
            or meta.get("url")
        )
        if not page_url:
            continue

        markdown = page.get("markdown") or page.get("content") or ""
        title = meta.get("title") or page_url
        domain = urlparse(page_url).netloc

        chunks = chunk_text(markdown)

        for i, chunk in enumerate(chunks):
            vector = embedder.embed_text(chunk)
            docs.append(
                {
                    "id": sha_id(page_url, str(i)),
                    "title": title,
                    "url": page_url,
                    "content": chunk,
                    "domain": domain,
                    "chunk_index": i,
                    "content_vector": vector,
                }
            )

    return docs
