# api/main.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import shutil

from orchestration.pipeline import run_pipeline
from qdrant_client import QdrantClient
import ollama

# ── App Setup ────────────────────────────────────────────────────
app = FastAPI(
    title="JEE RAG API",
    description="Retrieval Augmented Generation for JEE/NCERT preparation",
    version="1.0.0"
)

# CORS — allows your frontend (running on a different port) to call this API
# Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production, restrict to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    subject:  Optional[str] = None    # optional override — frontend can pass "Physics"

class SourceModel(BaseModel):
    chapter: str
    page:    int
    source:  str

class QueryResponse(BaseModel):
    question:          str
    concept:           Optional[str]
    explanation:       Optional[str]
    formula:           Optional[str]
    key_insight:       Optional[str]
    confidence:        str
    sources:           list
    validation:        Optional[dict]
    collection_used:   Optional[str]   # which Qdrant collection was searched
    error:             Optional[str]
    processing_time_ms: int


# ── Endpoint 1: Health Check ─────────────────────────────────────
@app.get("/health")
def health_check():
    """
    Checks if all three services are running:
    Qdrant, Ollama, and the API itself.
    Called by monitoring tools and the frontend on startup.
    """
    status = {
        "api":    "ok",
        "qdrant": "unknown",
        "ollama": "unknown"
    }

    # Check Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {str(e)}"

    # Check Ollama
    try:
        ollama.embeddings(model="nomic-embed-text", prompt="test")
        status["ollama"] = "ok"
    except Exception as e:
        status["ollama"] = f"error: {str(e)}"

    # Overall health
    all_ok = all(v == "ok" for v in status.values())
    status["status"] = "healthy" if all_ok else "degraded"

    return status


# ── Endpoint 2: Stats ────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    """
    Returns per-subject chunk counts from the three subject collections.
    Returns subject names as keys (not raw collection names) so the frontend
    can display "Physics: 2400 chunks" without knowing the internal naming.
    """
    try:
        client = QdrantClient(host="localhost", port=6333)

        # Query each subject collection directly rather than listing all
        # collections — this guarantees the response shape is always
        # {Physics, Chemistry, Maths} even if other collections exist in Qdrant.
        subject_map = {
            "Physics":   "physics_chunks",
            "Chemistry": "chemistry_chunks",
            "Maths":     "maths_chunks",
        }

        stats = {}
        total = 0
        for subject, collection_name in subject_map.items():
            try:
                count = client.count(collection_name=collection_name).count
            except Exception:
                count = 0   # collection not yet created (e.g. ingestion not run)
            stats[subject] = {"collection": collection_name, "chunks": count}
            total += count

        return {"collections": stats, "total_chunks": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 3: Query ────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG endpoint. Takes a question, runs the full pipeline,
    returns a structured answer with citations and validation report.

    This is what your Streamlit frontend will call.
    """
    start_time = time.time()

    try:
        # Run the full LangGraph pipeline
        result = run_pipeline(request.question)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Pipeline succeeded with an answer
        if result["answer"]:
            answer = result["answer"]

            # Convert Pydantic source objects to dicts for JSON serialisation
            sources = [
                {"chapter": s.chapter, "page": s.page, "source": s.source}
                for s in answer.sources
            ]

            return QueryResponse(
                question           = request.question,
                concept            = answer.concept,
                explanation        = answer.explanation,
                formula            = answer.formula,
                key_insight        = answer.key_insight,
                confidence         = answer.confidence,
                sources            = sources,
                validation         = result.get("validation_report"),
                collection_used    = result.get("collection_used"),
                error              = result.get("error"),
                processing_time_ms = elapsed_ms
            )

        # Pipeline ran but no answer (out of scope, low confidence etc.)
        else:
            return QueryResponse(
                question           = request.question,
                concept            = None,
                explanation        = None,
                formula            = None,
                key_insight        = None,
                confidence         = "low",
                sources            = [],
                validation         = None,
                collection_used    = result.get("collection_used"),
                error              = result.get("error", "No answer generated"),
                processing_time_ms = elapsed_ms
            )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        raise HTTPException(
            status_code = 500,
            detail      = f"Pipeline error: {str(e)}"
        )


# ── Endpoint 4: Ingest ───────────────────────────────────────────
@app.post("/ingest")
async def ingest_pdf(
    file:    UploadFile = File(...),
    subject: str        = "Physics",
    chapter: str        = "Unknown"
):
    """
    Upload a PDF and add it to the knowledge base.
    Saves the file, runs the ingestion pipeline, returns chunk count.

    This lets you add new content without restarting the server.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Save uploaded file to data directory
    save_path = f"data/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run ingestion pipeline
        from ingestion.ingest import parse_pdf, chunk_text, index_chunks
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(host="localhost", port=6333)

        if not client.collection_exists("jee_chunks"):
            client.create_collection(
                collection_name="jee_chunks",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        # Temporarily override config for this upload
        import ingestion.ingest as ingest_module
        original_chapter = ingest_module.CHAPTER
        original_subject = ingest_module.SUBJECT
        original_path    = ingest_module.PDF_PATH

        ingest_module.CHAPTER  = chapter
        ingest_module.SUBJECT  = subject
        ingest_module.PDF_PATH = save_path

        pages  = parse_pdf(save_path)
        chunks = chunk_text(pages)
        index_chunks(chunks, client, "jee_chunks")

        # Restore original config
        ingest_module.CHAPTER  = original_chapter
        ingest_module.SUBJECT  = original_subject
        ingest_module.PDF_PATH = original_path

        return {
            "status":   "success",
            "filename": file.filename,
            "subject":  subject,
            "chapter":  chapter,
            "chunks_added": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)