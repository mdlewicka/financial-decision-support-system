import json
from pathlib import Path

from pypdf import PdfReader

from backend.src.rag.models import Document


def load_text_document(path: Path, document_id: str | None = None) -> Document:
    content = path.read_text(encoding="utf-8")
    return Document(
        document_id=document_id or path.stem,
        content=content,
        source=str(path),
        metadata={"title": path.stem, "file_type": path.suffix.lower()},
    )


def load_pdf_document(path: Path, document_id: str | None = None) -> Document:
    pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    return Document(
        document_id=document_id or path.stem,
        content="\n\n".join(pages),
        source=str(path),
        metadata={"title": path.stem, "file_type": ".pdf"},
    )


def load_document_directory(directory: Path) -> list[Document]:
    loaders = {
        ".md": load_text_document,
        ".pdf": load_pdf_document,
        ".txt": load_text_document,
    }
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        loader = loaders.get(path.suffix.lower())
        if path.is_file() and loader is not None:
            documents.append(loader(path))
    if not documents:
        raise ValueError(f"No supported documents found in {directory}.")
    return documents


def load_reference_corpus(path: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            document_id=item["document_id"],
            content=item["content"],
            source=item.get("source", ""),
            metadata=item.get("metadata", {}),
        )
        for item in payload["documents"]
    ]
