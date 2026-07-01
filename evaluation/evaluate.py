# evaluation/evaluate.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import ollama
from retrieval.retriever import hybrid_retrieve
from generation.generator import generate_answer
from evaluation.validator import cosine_similarity

# ── Test Dataset ─────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "What is work done by a force?",
        "ground_truth": "Work done by a force is the product of force and displacement. W = F.d.cosθ where θ is the angle between force and displacement."
    },
    {
        "question": "When is work done by a force zero?",
        "ground_truth": "Work done is zero when there is no displacement, or when force is perpendicular to displacement making cosθ zero."
    },
    {
        "question": "What is kinetic energy?",
        "ground_truth": "Kinetic energy is energy due to motion. KE = 0.5 × m × v² where m is mass and v is velocity."
    },
    {
        "question": "What is the work energy theorem?",
        "ground_truth": "The work energy theorem states that change in kinetic energy equals work done by net force. Kf - Ki = W."
    },
    {
        "question": "What is potential energy?",
        "ground_truth": "Potential energy is stored energy due to position. Gravitational PE = mgh."
    }
]


# ── Metric 1: Faithfulness ───────────────────────────────────────
def score_faithfulness(answer_text, chunk_texts):
    """
    Asks Mistral: which claims in the answer are supported by the context?
    Score = supported claims / total claims
    """
    context = "\n\n".join(chunk_texts)

    prompt = f"""Given this context:
{context}

And this answer:
{answer_text}

List each factual claim in the answer. For each claim, state if it is 
supported by the context or not.
Respond in JSON format only:
{{
  "claims": [
    {{"claim": "claim text", "supported": true}},
    {{"claim": "claim text", "supported": false}}
  ]
}}"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw = response["message"]["content"]

        # Extract JSON
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        data  = json.loads(raw[start:end])

        claims     = data.get("claims", [])
        if not claims:
            return 0.0

        supported  = sum(1 for c in claims if c.get("supported", False))
        score      = supported / len(claims)
        return round(score, 3)

    except Exception as e:
        print(f"    Faithfulness error: {e}")
        return 0.0


# ── Metric 2: Answer Relevancy ───────────────────────────────────
def score_answer_relevancy(question, answer_text):
    """
    Generates questions from the answer, then measures how similar
    those generated questions are to the original question.
    High similarity = answer addressed the right question.
    """
    prompt = f"""Given this answer:
{answer_text}

Generate 3 questions that this answer would be responding to.
Respond in JSON format only:
{{"questions": ["question 1", "question 2", "question 3"]}}"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw  = response["message"]["content"]
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        data  = json.loads(raw[start:end])

        generated_questions = data.get("questions", [])
        if not generated_questions:
            return 0.0

        # Embed original question
        orig_emb = ollama.embeddings(
            model="nomic-embed-text",
            prompt=question
        )["embedding"]

        # Embed each generated question and compute similarity
        similarities = []
        for gq in generated_questions:
            gq_emb = ollama.embeddings(
                model="nomic-embed-text",
                prompt=gq
            )["embedding"]
            sim = cosine_similarity(orig_emb, gq_emb)
            similarities.append(sim)

        score = np.mean(similarities)
        return round(float(score), 3)

    except Exception as e:
        print(f"    Relevancy error: {e}")
        return 0.0


# ── Metric 3: Context Precision ──────────────────────────────────
def score_context_precision(question, chunk_texts):
    """
    For each retrieved chunk, asks: is this chunk relevant to the question?
    Precision = relevant chunks / total chunks
    Penalises irrelevant chunks appearing early in the ranking.
    """
    relevant_count = 0
    total          = len(chunk_texts)

    for chunk in chunk_texts:
        prompt = f"""Question: {question}

Context chunk:
{chunk[:500]}

Is this chunk relevant to answering the question? 
Reply with JSON only: {{"relevant": true}} or {{"relevant": false}}"""

        try:
            response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw   = response["message"]["content"]
            start = raw.find('{')
            end   = raw.rfind('}') + 1
            data  = json.loads(raw[start:end])

            if data.get("relevant", False):
                relevant_count += 1

        except Exception:
            pass

    score = relevant_count / total if total > 0 else 0.0
    return round(score, 3)


# ── Metric 4: Context Recall ─────────────────────────────────────
def score_context_recall(ground_truth, chunk_texts):
    """
    Checks how many facts from the ground truth answer
    are present in the retrieved chunks.
    Recall = facts found in context / total facts in ground truth
    """
    context = "\n\n".join(chunk_texts)

    prompt = f"""Ground truth answer:
{ground_truth}

Retrieved context:
{context}

Break the ground truth into individual facts.
For each fact, check if it is present in the retrieved context.
Respond in JSON only:
{{
  "facts": [
    {{"fact": "fact text", "found_in_context": true}},
    {{"fact": "fact text", "found_in_context": false}}
  ]
}}"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw   = response["message"]["content"]
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        data  = json.loads(raw[start:end])

        facts = data.get("facts", [])
        if not facts:
            return 0.0

        found = sum(1 for f in facts if f.get("found_in_context", False))
        score = found / len(facts)
        return round(score, 3)

    except Exception as e:
        print(f"    Recall error: {e}")
        return 0.0


# ── Run Evaluation ───────────────────────────────────────────────
def run_evaluation():
    print("="*50)
    print("JEE RAG EVALUATION")
    print("="*50)
    print(f"Running {len(TEST_CASES)} test cases...\n")

    all_scores = []

    for i, test in enumerate(TEST_CASES):
        print(f"Test {i+1}/{len(TEST_CASES)}: {test['question'][:50]}...")

        try:
            # Run pipeline
            retrieved   = hybrid_retrieve(test["question"], subject_filter="Physics")
            answer      = generate_answer(test["question"], retrieved)
            chunk_texts = [data["payload"]["text"] for _, data in retrieved]

            # Score all four metrics
            print("    Scoring faithfulness...")
            faith = score_faithfulness(answer.explanation, chunk_texts)

            print("    Scoring relevancy...")
            rel   = score_answer_relevancy(test["question"], answer.explanation)

            print("    Scoring precision...")
            prec  = score_context_precision(test["question"], chunk_texts)

            print("    Scoring recall...")
            rec   = score_context_recall(test["ground_truth"], chunk_texts)

            scores = {
                "question":    test["question"],
                "faithfulness": faith,
                "relevancy":   rel,
                "precision":   prec,
                "recall":      rec,
                "answer":      answer.explanation[:100]
            }
            all_scores.append(scores)

            print(f"    ✓ F={faith} R={rel} P={prec} Rec={rec}")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            all_scores.append({
                "question":     test["question"],
                "faithfulness": 0.0,
                "relevancy":    0.0,
                "precision":    0.0,
                "recall":       0.0,
                "answer":       "failed"
            })

    # ── Print Report ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)

    avg_faith = np.mean([s["faithfulness"] for s in all_scores])
    avg_rel   = np.mean([s["relevancy"]    for s in all_scores])
    avg_prec  = np.mean([s["precision"]    for s in all_scores])
    avg_rec   = np.mean([s["recall"]       for s in all_scores])

    print(f"\nOverall Scores:")
    print(f"  Faithfulness      : {avg_faith:.3f}  {'✓' if avg_faith >= 0.7 else '⚠'}")
    print(f"  Answer Relevancy  : {avg_rel:.3f}  {'✓' if avg_rel >= 0.7 else '⚠'}")
    print(f"  Context Precision : {avg_prec:.3f}  {'✓' if avg_prec >= 0.7 else '⚠'}")
    print(f"  Context Recall    : {avg_rec:.3f}  {'✓' if avg_rec >= 0.7 else '⚠'}")

    print(f"\nPer-question breakdown:")
    for s in all_scores:
        print(f"\n  Q: {s['question'][:50]}...")
        print(f"    Faithfulness : {s['faithfulness']}")
        print(f"    Relevancy    : {s['relevancy']}")
        print(f"    Precision    : {s['precision']}")
        print(f"    Recall       : {s['recall']}")

    # Save results
    import csv
    with open("evaluation/eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question","faithfulness","relevancy","precision","recall","answer"])
        writer.writeheader()
        writer.writerows(all_scores)

    print(f"\n✓ Results saved to evaluation/eval_results.csv")
    return all_scores


if __name__ == "__main__":
    run_evaluation()