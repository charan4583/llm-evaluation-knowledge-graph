"""Page 2 — LLM Response Evaluation with quantitative metrics, claim analysis, and optional AI Judge."""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from src.ai.ai_judge import evaluate_with_ai_judge
from src.ai.hallucination_analysis import CLAIM_DISCLAIMER, analyze_claims_rule_based
from src.database.connection import get_session, init_db
from src.database.repository import EvaluationRepository
from src.evaluation.evaluator import evaluate_response
from src.nlp.entity_extraction import extract_entities
from src.nlp.relationship_extraction import extract_relationships
from src.ui.theme import apply_theme, render_page_header

st.set_page_config(page_title="Evaluate Response | LLM Eval & KG", page_icon="🧪", layout="wide")
init_db()
apply_theme()

render_page_header(
    "Evaluate LLM Response",
    "Compare an AI-generated answer against a reference answer using lexical metrics, embeddings, claim analysis, and optional Gemini AI Judge.",
)

for key in ("eval_question", "eval_reference", "eval_generated"):
    if key not in st.session_state:
        st.session_state[key] = ""

with st.form("evaluate_form"):
    question = st.text_area("Question / Prompt", key="eval_question", height=80, placeholder="e.g., What is Python?")
    col_a, col_b = st.columns(2)
    with col_a:
        reference_answer = st.text_area(
            "Reference Answer (Ground Truth)",
            key="eval_reference",
            height=170,
            placeholder="Paste the expected reference answer...",
        )
    with col_b:
        generated_answer = st.text_area(
            "AI-Generated Answer",
            key="eval_generated",
            height=170,
            placeholder="Paste the LLM-generated answer to evaluate...",
        )

    use_ai_judge = st.checkbox(
        "Include Gemini AI Judge evaluation (optional)",
        value=False,
        disabled=not settings.is_gemini_configured,
        help="Requires GEMINI_API_KEY in .env. Avoid submitting confidential text to external APIs.",
    )
    if use_ai_judge:
        st.warning("Privacy notice: Enabling AI Judge sends the question and answers to the Gemini API.")

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    submitted = btn_col1.form_submit_button("Evaluate", type="primary", use_container_width=True)
    cleared = btn_col2.form_submit_button("Clear", use_container_width=True)

if cleared:
    st.session_state.pop("last_eval_bundle", None)
    st.rerun()

if submitted:
    if not reference_answer.strip() or not generated_answer.strip():
        st.error("Please provide both a Reference Answer and an AI-Generated Answer.")
    else:
        with st.spinner("Running evaluation pipeline..."):
            eval_res = evaluate_response(
                question=question or "Untitled Question",
                reference_answer=reference_answer,
                generated_answer=generated_answer,
                include_semantic=True,
            )
            claims = analyze_claims_rule_based(reference_answer, generated_answer)
            combined_text = f"{reference_answer}\n{generated_answer}"
            entities = extract_entities(combined_text)
            relationships = extract_relationships(combined_text, entities)

            ai_output_dict = None
            ai_error = None
            if use_ai_judge:
                ai_obj, ai_error = evaluate_with_ai_judge(
                    question or "Untitled Question",
                    reference_answer,
                    generated_answer,
                )
                if ai_obj is not None:
                    ai_output_dict = ai_obj.model_dump()

            st.session_state["last_eval_bundle"] = {
                "result": eval_res,
                "claims": claims,
                "entities": entities,
                "relationships": relationships,
                "ai_output": ai_output_dict,
                "ai_error": ai_error,
            }

bundle = st.session_state.get("last_eval_bundle")
if bundle:
    res = bundle["result"]
    claims = bundle["claims"]
    entities = bundle["entities"]
    relationships = bundle["relationships"]
    ai_output = bundle["ai_output"]
    ai_error = bundle["ai_error"]

    st.divider()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Precision", f"{res.precision:.1%}", help="TP / (TP + FP). Share of generated content tokens matching the reference.")
    m2.metric("Recall", f"{res.recall:.1%}", help="TP / (TP + FN). Share of reference content tokens captured.")
    m3.metric("F1-Score", f"{res.f1_score:.1%}", help="Harmonic mean of Precision and Recall.")
    m4.metric("Exact Match", f"{res.exact_match:.0%}", help="100% if normalized strings are identical, else 0%.")
    m5.metric("Semantic Similarity", f"{res.semantic_similarity:.1%}", help="Cosine similarity of all-MiniLM-L6-v2 embeddings.")
    m6.metric("Keyword Coverage", f"{res.keyword_coverage:.1%}", help="Fraction of key reference terms present in generated answer.")

    if res.semantic_details.get("error"):
        st.warning(f"Semantic similarity warning: {res.semantic_details['error']}")
    else:
        st.info(
            f"**Semantic Interpretation:** {res.semantic_details.get('interpretation', '')} "
            f"*(Note: {res.semantic_details.get('disclaimer', '')})*"
        )

    save_col, _ = st.columns([1.5, 4.5])
    if save_col.button("Save Evaluation to History", type="secondary", use_container_width=True):
        with get_session() as session:
            repo = EvaluationRepository(session)
            saved = repo.create_evaluation(
                question=res.question,
                reference_answer=res.reference_answer,
                generated_answer=res.generated_answer,
                precision=res.precision,
                recall=res.recall,
                f1_score=res.f1_score,
                exact_match=res.exact_match,
                semantic_similarity=res.semantic_similarity,
                keyword_coverage=res.keyword_coverage,
                ai_score=ai_output.get("overall_score") if ai_output else None,
                ai_feedback=ai_output,
                metadata={"missing_information": res.missing_information, "claim_analysis": claims},
                entities=entities,
                relationships=relationships,
            )
            st.success(f"Saved evaluation #{saved.id} to SQLite database.")

    with st.expander("Metric Calculation Breakdown & Limitations", expanded=False):
        prf = res.prf_details
        st.write(
            f"- **True Positives (TP):** {prf.get('true_positives', 0)} | "
            f"**False Positives (FP):** {prf.get('false_positives', 0)} | "
            f"**False Negatives (FN):** {prf.get('false_negatives', 0)}"
        )
        st.write(
            "**Formulae:** `Precision = TP / (TP + FP)`, `Recall = TP / (TP + FN)`, "
            "`F1 = 2 * Precision * Recall / (Precision + Recall)`"
        )
        st.write(
            "**Limitations:** Lexical token overlap penalizes valid synonyms and paraphrasing. "
            "Neither lexical overlap nor semantic similarity proves universal factual truth."
        )
        st.write(f"**Matched tokens:** {', '.join(prf.get('matched_tokens', [])) or 'None'}")
        st.write(f"**Missing reference tokens:** {', '.join(prf.get('missing_tokens', [])) or 'None'}")
        st.write(f"**Additional generated tokens:** {', '.join(prf.get('additional_tokens', [])) or 'None'}")

    tab_kw, tab_claims, tab_ai, tab_kg = st.tabs(
        ["Keyword & Gap Analysis", "Claim Verification", "AI Judge (Optional)", "Extracted Entities & Relations"]
    )

    with tab_kw:
        kw = res.keyword_details
        kc1, kc2, kc3 = st.columns(3)
        kc1.write("**Reference Keywords**")
        kc1.write(", ".join(kw.get("reference_keywords", [])) or "None")
        kc2.write("**Matched Keywords**")
        kc2.write(", ".join(kw.get("matched_keywords", [])) or "None")
        kc3.write("**Missing Keywords**")
        kc3.write(", ".join(kw.get("missing_keywords", [])) or "None")

        st.subheader("Missing Information")
        if res.missing_information:
            for item in res.missing_information:
                st.write(f"- {item}")
        else:
            st.success("No major reference gaps detected.")

        st.subheader("Additional Information")
        if res.additional_information:
            for item in res.additional_information:
                st.write(f"- {item}")
        else:
            st.write("No additional terms outside the reference.")

    with tab_claims:
        st.caption(CLAIM_DISCLAIMER)
        if not claims:
            st.info("No sentences available for claim analysis.")
        else:
            for idx, c in enumerate(claims, 1):
                cls = c["classification"]
                badge_cls = {
                    "Supported": "badge-success",
                    "Partially supported": "badge-info",
                    "Requires verification": "badge-warn",
                    "Not found in reference": "badge-danger",
                }.get(cls, "")
                conf_txt = f" | Confidence: {c['confidence']:.0%}" if c.get("confidence") is not None else ""
                st.markdown(
                    f"""
                    <div class="app-card">
                        <span class="badge {badge_cls}">{cls}</span>
                        <span class="app-muted">Claim {idx}{conf_txt}</span>
                        <p style="margin-top:8px;"><b>{c['claim_text']}</b></p>
                        <p class="app-muted">{c['explanation']}</p>
                        {"<p class='app-muted'><i>Reference evidence:</i> " + c['evidence'] + "</p>" if c.get('evidence') else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_ai:
        if ai_error:
            st.error(ai_error)
        elif ai_output:
            st.info(ai_output.get("disclaimer", ""))
            ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
            ac1.metric("Overall", f"{ai_output['overall_score']:.0f}/100")
            ac2.metric("Correctness", f"{ai_output['correctness']:.0f}/100")
            ac3.metric("Relevance", f"{ai_output['relevance']:.0f}/100")
            ac4.metric("Completeness", f"{ai_output['completeness']:.0f}/100")
            ac5.metric("Clarity", f"{ai_output['clarity']:.0f}/100")
            ac6.metric("Consistency", f"{ai_output['consistency']:.0f}/100")
            st.write(f"**Verdict:** {ai_output['verdict']}")
            st.write(f"**Explanation:** {ai_output['explanation']}")
            st.write("**Strengths:**", ai_output["strengths"])
            st.write("**Missing Information:**", ai_output["missing_information"])
            st.write("**Unsupported Claims:**", ai_output["unsupported_claims"])
            st.write("**Improvement Suggestions:**", ai_output["improvement_suggestions"])
        else:
            st.info("Enable the Gemini AI Judge checkbox before evaluating to view AI-assisted feedback.")

    with tab_kg:
        kg_c1, kg_c2 = st.columns(2)
        with kg_c1:
            st.write(f"**Extracted Entities ({len(entities)})**")
            st.dataframe(entities, use_container_width=True, hide_index=True)
        with kg_c2:
            st.write(f"**Extracted Relationships ({len(relationships)})**")
            st.dataframe(relationships, use_container_width=True, hide_index=True)
