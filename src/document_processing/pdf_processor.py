"""PDF text extraction using PyMuPDF with scanned/empty PDF handling."""

from __future__ import annotations

import pymupdf


class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text safely from in-memory PDF bytes without executing embedded content."""
    if not file_bytes:
        raise PDFProcessingError("The uploaded PDF file is empty.")

    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise PDFProcessingError("The PDF contains 0 pages.")
            pages_text: list[str] = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    pages_text.append(text.strip())
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Could not read the PDF file (it may be corrupted or encrypted): {exc}") from exc

    combined = "\n\n".join(pages_text).strip()
    if not combined:
        raise PDFProcessingError(
            "No extractable text found in this PDF. If it is a scanned/image-based PDF, OCR is not supported in this version."
        )
    return combined
