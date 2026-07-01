# evaluation/validator.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import ollama
import numpy as np
from typing import List, Tuple
from generation.generator import JEEAnswer, SourceCitation


# ── Utility: Cosine Similarity ───────────────────────────────────
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Measures the angle between two vectors.
    Returns a value between 0 and 1.
    1.0 = identical direction = same meaning
    0.0 = perpendicular = unrelated meaning

    We implement this manually so you understand what's happening
    under the hood — no black box library needed.
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_product    = np.dot(a, b)           # how much they point in same direction
    magnitude_a    = np.linalg.norm(a)      # length of vector a
    magnitude_b    = np.linalg.norm(b)      # length of vector b

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ── Check 1: Grounding Check ─────────────────────────────────────
def check_grounding(answer: JEEAnswer, retrieved_chunks: List) -> Tuple[float, str]:
    """
    Verifies that the answer's explanation is semantically grounded
    in the retrieved chunks — not invented by the LLM.

    How it works:
    1. Embed the answer's explanation
    2. Embed each retrieved chunk
    3. Compute cosine similarity between answer and each chunk
    4. Take the maximum similarity score

    If the answer came from the chunks, it should be semantically
    close to at least one of them. If it didn't, similarity will be low.

    Threshold: 0.5 — below this, answer is likely hallucinated
    """
    GROUNDING_THRESHOLD = 0.5

    # Embed the answer explanation
    answer_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=answer.explanation
    )["embedding"]

    # Embed each chunk and compute similarity
    max_similarity = 0.0
    for chunk_id, data in retrieved_chunks:
        chunk_text      = data["payload"]["text"]
        chunk_embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk_text
        )["embedding"]

        similarity    = cosine_similarity(answer_embedding, chunk_embedding)
        max_similarity = max(max_similarity, similarity)

    passed = max_similarity >= GROUNDING_THRESHOLD
    status = "PASS" if passed else "FAIL"
    detail = f"Max similarity to source chunks: {max_similarity:.4f} (threshold: {GROUNDING_THRESHOLD})"

    return max_similarity, f"{status} — {detail}"


# ── Check 2: Formula Check ───────────────────────────────────────
def check_formula(answer: JEEAnswer) -> Tuple[bool, str]:
    """
    Validates that the formula field looks like an actual formula.
    Not a full mathematical parser — just sanity checks.

    Valid formula patterns:
    - Contains at least one variable (single letter)
    - Contains an operator (=, +, -, *, /, ^)
    - Not just a single word

    Examples:
    PASS: "W = F.d", "F = ma", "KE = 0.5mv²", "E = mc²"
    FAIL: "work", "force times distance", None
    """

    # No formula is fine for conceptual questions
    if answer.formula is None:
        return True, "PASS — No formula (conceptual question)"

    formula = answer.formula.strip()

    # Must contain an equals sign or mathematical operator
    has_operator = bool(re.search(r'[=+\-*/^]', formula))

    # Must contain at least one letter (variable)
    has_variable = bool(re.search(r'[a-zA-Z]', formula))

    # Must be reasonably short (formulas aren't paragraphs)
    reasonable_length = len(formula) < 100

    # Must not be just a plain English sentence
    not_plain_english = not bool(re.match(r'^[A-Z][a-z]+ [a-z]+ [a-z]+', formula))

    passed = has_operator and has_variable and reasonable_length
    status = "PASS" if passed else "FAIL"

    details = []
    if not has_operator: details.append("no mathematical operator found")
    if not has_variable: details.append("no variable found")
    if not reasonable_length: details.append("formula too long")

    detail = f"{status} — Formula: '{formula}'"
    if details:
        detail += f" | Issues: {', '.join(details)}"

    return passed, detail


# ── Check 3: Citation Check ──────────────────────────────────────
def check_citations(answer: JEEAnswer, retrieved_chunks: List) -> Tuple[bool, str]:
    """
    Verifies that the page numbers cited in the answer
    actually exist in our retrieved chunks.

    This catches a common hallucination pattern where the LLM
    invents plausible-sounding page numbers that don't exist
    in the source material.

    Cross-references:
    answer.sources[].page  →  retrieved_chunks[].payload["page"]
    """

    # Collect all real page numbers from retrieved chunks
    real_pages = set()
    for chunk_id, data in retrieved_chunks:
        real_pages.add(data["payload"]["page"])

    # Check each cited source
    invalid_citations = []
    for source in answer.sources:
        if source.page not in real_pages:
            invalid_citations.append(
                f"Page {source.page} not in retrieved chunks {sorted(real_pages)}"
            )

    passed = len(invalid_citations) == 0
    status = "PASS" if passed else "FAIL"

    if passed:
        detail = f"{status} — All {len(answer.sources)} citations verified against retrieved pages {sorted(real_pages)}"
    else:
        detail = f"{status} — Invalid citations: {'; '.join(invalid_citations)}"

    return passed, detail


# ── Main Validation Runner ───────────────────────────────────────
def validate(answer: JEEAnswer, retrieved_chunks: List) -> dict:
    """
    Runs all three checks and returns a validation report.
    The report is attached to the answer before returning to the student.
    """
    print("\n🔍 Running validation checks...")

    # Run all checks
    print("   Check 1: Grounding...")
    grounding_score, grounding_detail = check_grounding(answer, retrieved_chunks)

    print("   Check 2: Formula...")
    formula_valid, formula_detail = check_formula(answer)

    print("   Check 3: Citations...")
    citations_valid, citation_detail = check_citations(answer, retrieved_chunks)

    # Overall pass — all checks must pass
    overall_pass = (
        grounding_score >= 0.5 and
        formula_valid           and
        citations_valid
    )

    report = {
        "overall":         "PASS" if overall_pass else "FAIL",
        "grounding_score": round(grounding_score, 4),
        "grounding":       grounding_detail,
        "formula":         formula_detail,
        "citations":       citation_detail,
        "checks_passed":   sum([grounding_score >= 0.5, formula_valid, citations_valid]),
        "checks_total":    3
    }

    return report


# ── Test it ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval.retriever import hybrid_retrieve
    from generation.generator import generate_answer

    query = "What is work done by a force and when is it zero?"

    print("Step 1: Retrieving...")
    retrieved = hybrid_retrieve(query, subject_filter="Physics")

    print("Step 2: Generating...")
    answer = generate_answer(query, retrieved)

    print("Step 3: Validating...")
    report = validate(answer, retrieved)

    print("\n=== Validation Report ===")
    print(f"Overall      : {report['overall']}")
    print(f"Grounding    : {report['grounding']}")
    print(f"Formula      : {report['formula']}")
    print(f"Citations    : {report['citations']}")
    print(f"Score        : {report['checks_passed']}/{report['checks_total']} checks passed")

    print("\n=== Answer ===")
    print(f"Concept      : {answer.concept}")
    print(f"Formula      : {answer.formula}")
    print(f"Confidence   : {answer.confidence}")