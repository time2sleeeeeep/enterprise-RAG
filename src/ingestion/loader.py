from pathlib import Path
from dataclasses import dataclass, field

import fitz
from docx import Document as DocxDocument


@dataclass
class LoadedDocument:
    content: str
    filename: str
    file_type: str
    metadata: dict = field(default_factory=dict)


def load_pdf(file_path: Path) -> LoadedDocument:
    doc = fitz.open(str(file_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[PAGE {page_num}]\n{text}")
    doc.close()
    return LoadedDocument(
        content="\n\n".join(pages),
        filename=file_path.name,
        file_type="pdf",
        metadata={"page_count": len(pages)},
    )


def load_markdown(file_path: Path) -> LoadedDocument:
    content = file_path.read_text(encoding="utf-8")
    return LoadedDocument(
        content=content,
        filename=file_path.name,
        file_type="markdown",
    )


def load_docx(file_path: Path) -> LoadedDocument:
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return LoadedDocument(
        content="\n\n".join(paragraphs),
        filename=file_path.name,
        file_type="docx",
    )


LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".docx": load_docx,
}


def load_document(file_path: Path) -> LoadedDocument:
    suffix = file_path.suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(file_path)
