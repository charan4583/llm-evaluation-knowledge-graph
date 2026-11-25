"""Formatted PDF evaluation report generator using ReportLab."""

from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_p(text: Any, style: ParagraphStyle) -> Paragraph:
    import html

    cleaned = html.escape(str(text or "")).replace("\n", "<br/>")
    return Paragraph(cleaned, style)


def generate_pdf_report(report_data: dict[str, Any]) -> bytes:
    """Generate a clean, structured PDF evaluation report from actual evaluation data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="LLM Evaluation & Knowledge Graph Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    muted_style = ParagraphStyle(
        "MutedCustom",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story: list[Any] = []
    story.append(Paragraph("LLM Evaluation &amp; Knowledge Graph Explorer", title_style))
    story.append(
        _safe_p(
            f"Evaluation Report | Date: {report_data.get('created_at', 'N/A')} | Record ID: {report_data.get('id', 'Live')}",
            muted_style,
        )
    )
    story.append(Spacer(1, 10))

    # Question & Answers
    story.append(Paragraph("1. Question &amp; Answers", h2_style))
    story.append(_safe_p(f"<b>Question:</b> {report_data.get('question', '')}", body_style))
    story.append(Spacer(1, 4))
    story.append(_safe_p(f"<b>Reference Answer:</b> {report_data.get('reference_answer', '')}", body_style))
    story.append(Spacer(1, 4))
    story.append(_safe_p(f"<b>Generated Answer:</b> {report_data.get('generated_answer', '')}", body_style))

    # Quantitative Metrics
    story.append(Paragraph("2. Quantitative Evaluation Metrics", h2_style))
    metrics_rows = [
        ["Metric", "Score", "Notes"],
        ["Precision", f"{report_data.get('precision', 0.0):.2%}", "Fraction of generated content tokens present in reference"],
        ["Recall", f"{report_data.get('recall', 0.0):.2%}", "Fraction of reference content tokens captured"],
        ["F1-Score", f"{report_data.get('f1_score', 0.0):.2%}", "Harmonic mean of Precision and Recall"],
        ["Exact Match", f"{report_data.get('exact_match', 0.0):.0%}", "Normalized exact string equality"],
        ["Semantic Similarity", f"{report_data.get('semantic_similarity', 0.0):.2%}", "Cosine similarity of embeddings (meaning, not factual truth)"],
        ["Keyword Coverage", f"{report_data.get('keyword_coverage', 0.0):.2%}", "Reference key terms matched in generated answer"],
    ]
    if report_data.get("ai_score") is not None:
        metrics_rows.append(["AI Judge Score", f"{report_data['ai_score']:.1f} / 100", "AI-assisted qualitative score (advisory)"])

    table = Table(metrics_rows, colWidths=[1.6 * inch, 1.1 * inch, 4.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)

    # Missing & Unsupported Information
    missing = report_data.get("missing_information") or []
    claims = report_data.get("claim_analysis") or []
    if missing or claims:
        story.append(Paragraph("3. Missing Information &amp; Claim Analysis", h2_style))
        for item in missing[:10]:
            story.append(_safe_p(f"• {item}", body_style))
        for c in claims[:12]:
            story.append(
                _safe_p(
                    f"• [{c.get('classification', 'Unknown')}] {c.get('claim_text', '')}",
                    body_style,
                )
            )

    # AI Judge Feedback
    ai_fb = report_data.get("ai_feedback")
    if isinstance(ai_fb, dict) and ai_fb:
        story.append(Paragraph("4. AI-Assisted Evaluation (Gemini Judge)", h2_style))
        story.append(_safe_p(f"<b>Verdict:</b> {ai_fb.get('verdict', '')}", body_style))
        story.append(_safe_p(f"<b>Explanation:</b> {ai_fb.get('explanation', '')}", body_style))
        story.append(_safe_p("<i>Note: AI-assisted evaluation can make mistakes and is not absolute truth.</i>", muted_style))

    # Entities & Relationships
    entities = report_data.get("entities") or []
    relationships = report_data.get("relationships") or []
    if entities or relationships:
        story.append(Paragraph("5. Extracted Knowledge Graph Summary", h2_style))
        story.append(_safe_p(f"Entities extracted: {len(entities)} | Relationships extracted: {len(relationships)}", body_style))
        for ent in entities[:20]:
            story.append(
                _safe_p(
                    f"• {ent.get('entity_name', '')} ({ent.get('entity_type', 'Other')}, freq={ent.get('frequency', 1)})",
                    body_style,
                )
            )
        for rel in relationships[:20]:
            story.append(
                _safe_p(
                    f"• {rel.get('source_entity', '')} → {rel.get('relationship_type', '')} → {rel.get('target_entity', '')}",
                    body_style,
                )
            )

    doc.build(story)
    return buffer.getvalue()
