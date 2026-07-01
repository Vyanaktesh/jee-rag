# orchestration/pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from retrieval.retriever import hybrid_retrieve, COLLECTION_MAP, DEFAULT_COLLECTION
from generation.generator import generate_answer
from evaluation.validator import validate
import ollama

# ── State Definition ─────────────────────────────────────────────
class JEEState(TypedDict):
    query:             str
    subject:           Optional[str]
    chapter:           Optional[str]
    retrieved_chunks:  Optional[List]
    context_score:     Optional[float]
    answer:            Optional[Any]
    validation_report: Optional[dict]
    collection_used:   Optional[str]   # which Qdrant collection was searched
    error:             Optional[str]
    step_log:          List[str]


# ── Node 1: Classify ─────────────────────────────────────────────
def classify(state: JEEState) -> JEEState:
    query = state["query"].lower()
    log   = state.get("step_log", [])

    physics_keywords   = ["force", "work", "energy", "power", "motion",
                          "velocity", "acceleration", "momentum", "newton",
                          "friction", "gravity", "wave", "light", "optics",
                          "electric", "magnetic", "heat", "thermodynamics"]

    chemistry_keywords = ["atom", "molecule", "bond", "reaction", "acid",
                          "base", "organic", "element", "periodic", "electron",
                          "proton", "neutron", "mole", "solution", "equilibrium"]

    maths_keywords     = ["integrate", "derivative", "matrix", "vector",
                          "probability", "permutation", "combination", "limit",
                          "function", "trigonometry", "algebra", "geometry"]

    scores = {
        "Physics":   sum(1 for k in physics_keywords   if k in query),
        "Chemistry": sum(1 for k in chemistry_keywords if k in query),
        "Maths":     sum(1 for k in maths_keywords     if k in query),
    }

    detected_subject = max(scores, key=scores.get)
    if scores[detected_subject] == 0:
        detected_subject = "Physics"

    log.append(f"classify → detected subject: {detected_subject} (scores: {scores})")
    print(f"\n[Node: classify] Subject detected: {detected_subject}")

    return {**state, "subject": detected_subject, "step_log": log}


# ── Node 2: Retrieve ─────────────────────────────────────────────
def retrieve(state: JEEState) -> JEEState:
    log = state.get("step_log", [])
    print(f"\n[Node: retrieve] Running hybrid retrieval...")

    # Resolve the collection here too so we can store it in state and
    # surface it to the API response — useful for debugging and the frontend.
    collection = COLLECTION_MAP.get(state["subject"], DEFAULT_COLLECTION)

    chunks = hybrid_retrieve(state["query"], subject_filter=state["subject"])

    avg_score = sum(data["score"] for _, data in chunks) / len(chunks) if chunks else 0.0

    log.append(f"retrieve → {collection} — got {len(chunks)} chunks, avg score: {avg_score:.4f}")
    print(f"[Node: retrieve] Got {len(chunks)} chunks from '{collection}', avg score: {avg_score:.4f}")

    return {
        **state,
        "retrieved_chunks": chunks,
        "context_score":    avg_score,
        "collection_used":  collection,
        "step_log":         log,
    }


# ── Node 3: Validate Context ─────────────────────────────────────
def validate_context(state: JEEState) -> JEEState:
    log    = state.get("step_log", [])
    score  = state.get("context_score", 0)
    chunks = state.get("retrieved_chunks", [])

    MIN_SCORE  = 0.01
    MIN_CHUNKS = 2

    if score < MIN_SCORE or len(chunks) < MIN_CHUNKS:
        log.append(f"validate_context → FAILED (score={score:.4f}, chunks={len(chunks)})")
        print(f"[Node: validate_context] ❌ Context insufficient")
        return {**state, "error": "Insufficient context — query may be outside indexed content", "step_log": log}

    log.append(f"validate_context → PASSED (score={score:.4f}, chunks={len(chunks)})")
    print(f"[Node: validate_context] ✓ Context sufficient")
    return {**state, "step_log": log}


# ── Node 4: Generate ─────────────────────────────────────────────
def generate(state: JEEState) -> JEEState:
    log = state.get("step_log", [])
    print(f"\n[Node: generate] Generating structured answer...")

    try:
        answer = generate_answer(state["query"], state["retrieved_chunks"])
        log.append(f"generate → answer generated, confidence: {answer.confidence}")
        print(f"[Node: generate] ✓ Answer generated (confidence: {answer.confidence})")
        return {**state, "answer": answer, "step_log": log}
    except Exception as e:
        log.append(f"generate → FAILED: {str(e)}")
        print(f"[Node: generate] ❌ Generation failed: {str(e)}")
        return {**state, "error": "Generation failed — query may be outside indexed content", "step_log": log}


# ── Node 5: Run Validation ───────────────────────────────────────
def run_validation(state: JEEState) -> JEEState:
    log    = state.get("step_log", [])
    answer = state.get("answer")

    if not answer:
        return {**state}

    report = validate(answer, state["retrieved_chunks"])
    log.append(f"validation → {report['overall']} ({report['checks_passed']}/{report['checks_total']})")
    print(f"[Node: run_validation] {report['overall']} — {report['checks_passed']}/3 checks passed")

    error = None
    if report["overall"] == "FAIL":
        error = f"Validation failed — {report['grounding']}"

    return {**state, "validation_report": report, "error": error, "step_log": log}


# ── Node 6: Validate Answer ──────────────────────────────────────
def validate_answer(state: JEEState) -> JEEState:
    log    = state.get("step_log", [])
    answer = state.get("answer")

    if answer and answer.confidence == "low":
        log.append("validate_answer → confidence LOW, flagging")
        print(f"[Node: validate_answer] ⚠️  Low confidence answer")
        return {**state, "error": "Low confidence — answer may be incomplete", "step_log": log}

    log.append("validate_answer → PASSED")
    print(f"[Node: validate_answer] ✓ Answer validated")
    return {**state, "step_log": log}


# ── Node 7: Handle Insufficient ──────────────────────────────────
def handle_insufficient(state: JEEState) -> JEEState:
    print(f"\n[Node: handle_insufficient] Returning error response")
    return {**state, "answer": None, "step_log": state.get("step_log", []) + ["handle_insufficient → returned error"]}


# ── Routing Function ─────────────────────────────────────────────
def should_generate(state: JEEState) -> str:
    if state.get("error"):
        return "handle_insufficient"
    return "generate"


# ── Build the Graph ──────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(JEEState)

    graph.add_node("classify",            classify)
    graph.add_node("retrieve",            retrieve)
    graph.add_node("validate_context",    validate_context)
    graph.add_node("generate",            generate)
    graph.add_node("run_validation",      run_validation)
    graph.add_node("validate_answer",     validate_answer)
    graph.add_node("handle_insufficient", handle_insufficient)

    graph.set_entry_point("classify")
    graph.add_edge("classify",            "retrieve")
    graph.add_edge("retrieve",            "validate_context")

    graph.add_conditional_edges(
        "validate_context",
        should_generate,
        {
            "generate":            "generate",
            "handle_insufficient": "handle_insufficient"
        }
    )

    graph.add_edge("generate",            "run_validation")
    graph.add_edge("run_validation",      "validate_answer")
    graph.add_edge("validate_answer",     END)
    graph.add_edge("handle_insufficient", END)

    return graph.compile()


# ── Run Pipeline ─────────────────────────────────────────────────
def run_pipeline(query: str):
    pipeline = build_pipeline()

    initial_state = {
        "query":             query,
        "subject":           None,
        "chapter":           None,
        "retrieved_chunks":  None,
        "context_score":     None,
        "answer":            None,
        "validation_report": None,
        "collection_used":   None,   # populated by retrieve node
        "error":             None,
        "step_log":          []
    }

    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")

    final_state = pipeline.invoke(initial_state)
    return final_state


# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    result = run_pipeline("What is work done by a force and when is it zero?")

    if result["answer"]:
        print(f"\n{'='*50}")
        print("FINAL ANSWER")
        print(f"{'='*50}")
        print(f"Concept     : {result['answer'].concept}")
        print(f"Explanation : {result['answer'].explanation}")
        print(f"Formula     : {result['answer'].formula}")
        print(f"Key Insight : {result['answer'].key_insight}")
        print(f"Confidence  : {result['answer'].confidence}")
        print(f"Collection  : {result['collection_used']}")
        print(f"Sources     :")
        for src in result['answer'].sources:
            print(f"  - {src.source}, {src.chapter}, Page {src.page}")

        if result["validation_report"]:
            vr = result["validation_report"]
            print(f"\nValidation  : {vr['overall']} ({vr['checks_passed']}/3 checks passed)")
            print(f"Grounding   : {vr['grounding']}")
            print(f"Formula     : {vr['formula']}")
            print(f"Citations   : {vr['citations']}")
    else:
        print(f"\nError: {result['error']}")

    print(f"\nStep Log:")
    for step in result["step_log"]:
        print(f"  → {step}")

    print("\n" + "="*50)

    result2 = run_pipeline("What is the capital of France?")
    if result2["answer"]:
        print(f"Concept: {result2['answer'].concept}")
    else:
        print(f"Handled gracefully: {result2['error']}")
