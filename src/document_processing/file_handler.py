"""Unified file upload validator and text extractor for PDF, DOCX, and TXT."""

from __future__ import annotations

from pathlib import Path

from config.settings import settings
from src.document_processing.docx_processor import DOCXProcessingError, extract_text_from_docx_bytes
from src.document_processing.pdf_processor import PDFProcessingError, extract_text_from_pdf_bytes

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class FileValidationError(RuntimeError):
    """Raised when an uploaded file fails validation or text extraction."""


def extract_text_from_upload(
    filename: str,
    file_bytes: bytes,
    *,
    max_size_mb: int | None = None,
) -> str:
    """Validate file type and size, then extract plain text safely."""
    if not filename:
        raise FileValidationError("Missing filename.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    limit_mb = max_size_mb or settings.max_upload_size_mb
    max_bytes = limit_mb * 1024 * 1024
    if not file_bytes:
        raise FileValidationError("Uploaded file is empty (0 bytes).")
    if len(file_bytes) > max_bytes:
        raise FileValidationError(
            f"File exceeds the maximum allowed size of {limit_mb} MB."
        )

    if ext == ".txt":
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
        if not text.strip():
            raise FileValidationError("The uploaded TXT file is empty.")
        return text.strip()

    if ext == ".pdf":
        try:
            return extract_text_from_pdf_bytes(file_bytes)
        except PDFProcessingError as exc:
            raise FileValidationError(str(exc)) from exc

    if ext == ".docx":
        try:
            return extract_text_from_docx_bytes(file_bytes)
        except DOCXProcessingError as exc:
            raise FileValidationError(str(exc)) from exc

    raise FileValidationError(f"Unsupported file extension: {ext}")
