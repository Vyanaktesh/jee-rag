# generation/generator.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from typing import List, Optional
import ollama
import json
import re

# ── Output Schema ────────────────────────────────────────────────
class SourceCitation(BaseModel):
    chapter: str
    page: int
    source: str

class JEEAnswer(BaseModel):
    concept: str
    explanation: str
    formula: Optional[str] = None
    key_insight: Optional[str] = None      # ← make this Optional
    step_by_step: Optional[List[str]] = None
    sources: List[SourceCitation]
    confidence: str

# ── Build Prompt ─────────────────────────────────────────────────
def build_prompt(query, retrieved_chunks):
    context_parts = []
    for i, (chunk_id, data) in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1} — {data['payload']['chapter']}, "
            f"Page {data['payload']['page']}]\n"
            f"{data['payload']['text']}"
        )
    context = "\n\n".join(context_parts)

    # Key change — we give Mistral an EXAMPLE of the exact JSON format
    # Small local models respond much better to examples than abstract schemas
    prompt = f"""You are an expert JEE/NCERT Physics teacher.
Answer the student's question using ONLY the provided context.
Do not use any knowledge outside the provided context.
If the context doesn't contain enough information, set confidence to "low".

CONTEXT:
{context}

STUDENT QUESTION:
{query}

Respond ONLY with a JSON object in exactly this format, no other text:
{{
  "concept": "one line description of the core concept",
  "explanation": "clear explanation in 2-3 sentences",
  "formula": "key formula as plain text, or null if not applicable",
  "key_insight": "most important thing to remember for JEE",
  "step_by_step": ["step 1", "step 2"] or null if not a calculation,
  "sources": [
    {{"chapter": "chapter name", "page": 3, "source": "NCERT Class 11"}}
  ],
  "confidence": "high or medium or low"
}}"""

    return prompt


# ── Parse Response ───────────────────────────────────────────────
def parse_response(raw_text):
    """
    Extracts JSON from Mistral's response and validates it
    against our Pydantic schema.
    """
    # Remove markdown code blocks if present
    raw_text = re.sub(r'```json\s*', '', raw_text)
    raw_text = re.sub(r'```\s*', '', raw_text)
    raw_text = raw_text.strip()

    # Find JSON object in the response
    start = raw_text.find('{')
    end   = raw_text.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON found in response")

    json_str = raw_text[start:end]
    data = json.loads(json_str)

    # Validate with Pydantic
    answer = JEEAnswer(**data)
    return answer


# ── Generate Answer ──────────────────────────────────────────────
def generate_answer(query, retrieved_chunks):
    print(f"\n🤖 Generating structured answer...")

    prompt = build_prompt(query, retrieved_chunks)

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}  # low temperature = more consistent output
    )

    raw_text = response["message"]["content"]

    # Parse and validate
    answer = parse_response(raw_text)
    return answer


# ── Test it ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval.retriever import hybrid_retrieve

    query = "What is work done by a force and when is it zero?"

    print("Step 1: Retrieving relevant chunks...")
    retrieved = hybrid_retrieve(query, subject_filter="Physics")

    print("Step 2: Generating structured answer...")
    answer = generate_answer(query, retrieved)

    print("\n=== Structured JEE Answer ===")
    print(f"\nConcept      : {answer.concept}")
    print(f"\nExplanation  : {answer.explanation}")
    print(f"\nFormula      : {answer.formula}")
    print(f"\nKey Insight  : {answer.key_insight}")

    if answer.step_by_step:
        print(f"\nStep by Step :")
        for i, step in enumerate(answer.step_by_step):
            print(f"  {i+1}. {step}")

    print(f"\nConfidence   : {answer.confidence}")
    print(f"\nSources      :")
    for src in answer.sources:
        print(f"  - {src.source}, {src.chapter}, Page {src.page}")