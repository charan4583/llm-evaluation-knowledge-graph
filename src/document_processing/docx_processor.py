"""DOCX text extraction using python-docx."""

from __future__ import annotations

from io import BytesIO

from docx import Document


class DOCXProcessingError(RuntimeError):
    """Raised when a DOCX file cannot be parsed or contains no text."""


def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    """Extract paragraph and table text safely from in-memory DOCX bytes."""
    if not file_bytes:
        raise DOCXProcessingError("The uploaded DOCX file is empty.")

    try:
        doc = Document(BytesIO(file_bytes))
        chunks: list[str] = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    chunks.append(row_text)
    except Exception as exc:
        raise DOCXProcessingError(f"Could not read the DOCX file (it may be corrupted): {exc}") from exc

    combined = "\n".join(chunks).strip()
    if not combined:
        raise DOCXProcessingError("The uploaded DOCX document contains no readable text.")
    return combined
