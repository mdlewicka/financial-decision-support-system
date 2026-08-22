from backend.src.rag.models import Chunk, Document


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(_split_document(document, chunk_size, chunk_overlap))
    return chunks


def _split_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    text = " ".join(document.content.split())
    parts: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        content = text[start:end].strip()
        if content:
            parts.append(
                Chunk(
                    chunk_id=f"{document.document_id}_chunk_{index:04d}",
                    document_id=document.document_id,
                    content=content,
                    source=document.source,
                    metadata=document.metadata,
                )
            )
            index += 1

        if end == len(text):
            break
        start = end - chunk_overlap

    return parts
