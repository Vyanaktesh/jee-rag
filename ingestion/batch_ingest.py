# ingestion/batch_ingest.py
#
# Batch ingestion pipeline for the full NCERT corpus.
# Downloads PDFs from ncert.nic.in, chunks them, embeds with nomic-embed-text,
# and stores in subject-specific Qdrant collections.
#
# Design principle: idempotent — safe to run multiple times.
# Already-indexed chapters are skipped, already-downloaded PDFs are not re-fetched.

import sys
import os

# Allow imports from project root (e.g. ingestion.ingest) regardless of
# where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import uuid
import argparse
import requests
from datetime import datetime

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

# Reuse the proven parse and chunk functions from single-chapter ingest.
# We do NOT reuse index_chunks because it hard-codes module-level SUBJECT/CHAPTER
# globals — we need per-chunk metadata control here.
from ingestion.ingest import parse_pdf, chunk_text

# ── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL  = "nomic-embed-text"
VECTOR_SIZE  = 768          # nomic-embed-text output dimension
NCERT_BASE   = "https://ncert.nic.in/textbook/pdf"
LOG_PATH     = "data/ingestion_log.json"
RETRY_COUNT  = 3
RETRY_DELAY  = 2            # seconds between download retries

# Map each subject to its dedicated Qdrant collection.
# Three collections instead of one because:
#   - Subject-scoped retrieval is faster (smaller search space)
#   - We can tune chunk size / embedding strategy per subject later
#   - The /query endpoint can route by subject without a filter overhead
COLLECTION_MAP = {
    "Physics":   "physics_chunks",
    "Chemistry": "chemistry_chunks",
    "Maths":     "maths_chunks",
}

# ── Corpus Definition ────────────────────────────────────────────────────────
# Each entry is one NCERT chapter PDF.  The filename matches the NCERT URL path.
# URL pattern: https://ncert.nic.in/textbook/pdf/{filename}

CORPUS = {
    "Physics": [
        # Class 11 Part 1 (keph1XX)
        {"filename": "keph101.pdf", "chapter": "Physical World",                          "class_level": "Class 11"},
        {"filename": "keph102.pdf", "chapter": "Units and Measurements",                  "class_level": "Class 11"},
        {"filename": "keph103.pdf", "chapter": "Motion in a Straight Line",               "class_level": "Class 11"},
        {"filename": "keph104.pdf", "chapter": "Motion in a Plane",                       "class_level": "Class 11"},
        {"filename": "keph105.pdf", "chapter": "Laws of Motion",                          "class_level": "Class 11"},
        {"filename": "keph106.pdf", "chapter": "Work Energy and Power",                   "class_level": "Class 11"},
        {"filename": "keph107.pdf", "chapter": "System of Particles and Rotational Motion","class_level": "Class 11"},
        {"filename": "keph108.pdf", "chapter": "Gravitation",                             "class_level": "Class 11"},
        # Class 11 Part 2 (keph2XX)
        {"filename": "keph201.pdf", "chapter": "Mechanical Properties of Solids",         "class_level": "Class 11"},
        {"filename": "keph202.pdf", "chapter": "Mechanical Properties of Fluids",         "class_level": "Class 11"},
        {"filename": "keph203.pdf", "chapter": "Thermal Properties of Matter",            "class_level": "Class 11"},
        {"filename": "keph204.pdf", "chapter": "Thermodynamics",                          "class_level": "Class 11"},
        {"filename": "keph205.pdf", "chapter": "Kinetic Theory",                          "class_level": "Class 11"},
        # Class 12 Part 1 (leph1XX)
        {"filename": "leph101.pdf", "chapter": "Electric Charges and Fields",             "class_level": "Class 12"},
        {"filename": "leph102.pdf", "chapter": "Electrostatic Potential",                 "class_level": "Class 12"},
        {"filename": "leph103.pdf", "chapter": "Current Electricity",                     "class_level": "Class 12"},
        {"filename": "leph104.pdf", "chapter": "Moving Charges and Magnetism",            "class_level": "Class 12"},
        {"filename": "leph105.pdf", "chapter": "Magnetism and Matter",                    "class_level": "Class 12"},
        {"filename": "leph106.pdf", "chapter": "Electromagnetic Induction",               "class_level": "Class 12"},
        {"filename": "leph107.pdf", "chapter": "Alternating Current",                     "class_level": "Class 12"},
        {"filename": "leph108.pdf", "chapter": "Electromagnetic Waves",                   "class_level": "Class 12"},
        # Class 12 Part 2 (leph2XX)
        {"filename": "leph201.pdf", "chapter": "Ray Optics",                              "class_level": "Class 12"},
        {"filename": "leph202.pdf", "chapter": "Wave Optics",                             "class_level": "Class 12"},
        {"filename": "leph203.pdf", "chapter": "Dual Nature of Radiation",                "class_level": "Class 12"},
        {"filename": "leph204.pdf", "chapter": "Atoms",                                   "class_level": "Class 12"},
        {"filename": "leph205.pdf", "chapter": "Nuclei",                                  "class_level": "Class 12"},
    ],

    "Chemistry": [
        # Class 11 Part 1 (kech1XX)
        {"filename": "kech101.pdf", "chapter": "Some Basic Concepts of Chemistry",        "class_level": "Class 11"},
        {"filename": "kech102.pdf", "chapter": "Structure of Atom",                       "class_level": "Class 11"},
        {"filename": "kech103.pdf", "chapter": "Classification of Elements",              "class_level": "Class 11"},
        {"filename": "kech104.pdf", "chapter": "Chemical Bonding",                        "class_level": "Class 11"},
        {"filename": "kech105.pdf", "chapter": "States of Matter",                        "class_level": "Class 11"},
        {"filename": "kech106.pdf", "chapter": "Thermodynamics",                          "class_level": "Class 11"},
        {"filename": "kech107.pdf", "chapter": "Equilibrium",                             "class_level": "Class 11"},
        # Class 11 Part 2 (kech2XX)
        {"filename": "kech201.pdf", "chapter": "Redox Reactions",                         "class_level": "Class 11"},
        {"filename": "kech202.pdf", "chapter": "Hydrogen",                                "class_level": "Class 11"},
        {"filename": "kech203.pdf", "chapter": "The s-Block Elements",                    "class_level": "Class 11"},
        {"filename": "kech204.pdf", "chapter": "Organic Chemistry",                       "class_level": "Class 11"},
        # Class 12 Part 1 (lech1XX)
        {"filename": "lech101.pdf", "chapter": "Solutions",                               "class_level": "Class 12"},
        {"filename": "lech102.pdf", "chapter": "Electrochemistry",                        "class_level": "Class 12"},
        {"filename": "lech103.pdf", "chapter": "Chemical Kinetics",                       "class_level": "Class 12"},
        {"filename": "lech104.pdf", "chapter": "d and f Block Elements",                  "class_level": "Class 12"},
        {"filename": "lech105.pdf", "chapter": "Coordination Compounds",                  "class_level": "Class 12"},
        {"filename": "lech106.pdf", "chapter": "Haloalkanes and Haloarenes",              "class_level": "Class 12"},
        {"filename": "lech107.pdf", "chapter": "Alcohols Phenols and Ethers",             "class_level": "Class 12"},
        # Class 12 Part 2 (lech2XX)
        {"filename": "lech201.pdf", "chapter": "Aldehydes Ketones and Carboxylic Acids",  "class_level": "Class 12"},
        {"filename": "lech202.pdf", "chapter": "Amines",                                  "class_level": "Class 12"},
        {"filename": "lech203.pdf", "chapter": "Biomolecules",                            "class_level": "Class 12"},
        {"filename": "lech204.pdf", "chapter": "Chemistry in Everyday Life",              "class_level": "Class 12"},
    ],

    "Maths": [
        # Class 11 (kemh1XX) — 16 chapters
        {"filename": "kemh101.pdf", "chapter": "Sets",                                    "class_level": "Class 11"},
        {"filename": "kemh102.pdf", "chapter": "Relations and Functions",                  "class_level": "Class 11"},
        {"filename": "kemh103.pdf", "chapter": "Trigonometric Functions",                 "class_level": "Class 11"},
        {"filename": "kemh104.pdf", "chapter": "Principle of Mathematical Induction",     "class_level": "Class 11"},
        {"filename": "kemh105.pdf", "chapter": "Complex Numbers",                         "class_level": "Class 11"},
        {"filename": "kemh106.pdf", "chapter": "Linear Inequalities",                     "class_level": "Class 11"},
        {"filename": "kemh107.pdf", "chapter": "Permutations and Combinations",           "class_level": "Class 11"},
        {"filename": "kemh108.pdf", "chapter": "Binomial Theorem",                        "class_level": "Class 11"},
        {"filename": "kemh109.pdf", "chapter": "Sequences and Series",                    "class_level": "Class 11"},
        {"filename": "kemh110.pdf", "chapter": "Straight Lines",                          "class_level": "Class 11"},
        {"filename": "kemh111.pdf", "chapter": "Conic Sections",                          "class_level": "Class 11"},
        {"filename": "kemh112.pdf", "chapter": "Introduction to 3D Geometry",             "class_level": "Class 11"},
        {"filename": "kemh113.pdf", "chapter": "Limits and Derivatives",                  "class_level": "Class 11"},
        {"filename": "kemh114.pdf", "chapter": "Mathematical Reasoning",                  "class_level": "Class 11"},
        {"filename": "kemh115.pdf", "chapter": "Statistics",                              "class_level": "Class 11"},
        {"filename": "kemh116.pdf", "chapter": "Probability",                             "class_level": "Class 11"},
        # Class 12 Part 1 (lemh1XX)
        {"filename": "lemh101.pdf", "chapter": "Relations and Functions",                  "class_level": "Class 12"},
        {"filename": "lemh102.pdf", "chapter": "Inverse Trigonometric Functions",         "class_level": "Class 12"},
        {"filename": "lemh103.pdf", "chapter": "Matrices",                                "class_level": "Class 12"},
        {"filename": "lemh104.pdf", "chapter": "Determinants",                            "class_level": "Class 12"},
        {"filename": "lemh105.pdf", "chapter": "Continuity and Differentiability",        "class_level": "Class 12"},
        {"filename": "lemh106.pdf", "chapter": "Application of Derivatives",              "class_level": "Class 12"},
        # Class 12 Part 2 (lemh2XX) — 7 chapters
        {"filename": "lemh201.pdf", "chapter": "Integrals",                               "class_level": "Class 12"},
        {"filename": "lemh202.pdf", "chapter": "Application of Integrals",                "class_level": "Class 12"},
        {"filename": "lemh203.pdf", "chapter": "Differential Equations",                  "class_level": "Class 12"},
        {"filename": "lemh204.pdf", "chapter": "Vector Algebra",                          "class_level": "Class 12"},
        {"filename": "lemh205.pdf", "chapter": "Three Dimensional Geometry",              "class_level": "Class 12"},
        {"filename": "lemh206.pdf", "chapter": "Linear Programming",                      "class_level": "Class 12"},
        {"filename": "lemh207.pdf", "chapter": "Probability",                             "class_level": "Class 12"},
    ],
}

# Attach URL to every entry now so callers don't have to build it themselves.
for _subject, _entries in CORPUS.items():
    for _entry in _entries:
        _entry["url"] = f"{NCERT_BASE}/{_entry['filename']}"


# ── Helper: timestamp string ──────────────────────────────────────────────────

def ts() -> str:
    """Return current time as HH:MM:SS for log prefixes."""
    return datetime.now().strftime("%H:%M:%S")


# ── Step 1: Download ─────────────────────────────────────────────────────────

def download_pdf(url: str, save_path: str) -> bool:
    """
    Download a PDF from `url` and save it to `save_path`.

    Skips the download if the file already exists (idempotent behaviour).
    Retries up to RETRY_COUNT times on network failure — transient errors
    (timeouts, rate-limits) are common when hitting government servers.

    Returns True on success (including already-exists), False after all retries fail.
    """
    if os.path.exists(save_path):
        print(f"[{ts()}]   ↩  Already downloaded: {os.path.basename(save_path)}")
        return True

    # Create subject subfolder (e.g. data/physics/) if it doesn't exist yet.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            print(f"[{ts()}]   ⬇  Downloading {url}  (attempt {attempt}/{RETRY_COUNT})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()          # raises on 4xx / 5xx
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"[{ts()}]   ✓  Saved to {save_path}")
            return True
        except Exception as e:
            print(f"[{ts()}]   ✗  Attempt {attempt} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)

    print(f"[{ts()}]   ✗  All retries exhausted for {url}")
    return False


# ── Step 2: Idempotency Check ─────────────────────────────────────────────────

def is_chapter_indexed(client: QdrantClient, collection_name: str, chapter_name: str) -> bool:
    """
    Check whether a chapter is already present in the Qdrant collection.

    Uses scroll with a payload filter rather than a vector search so we don't
    need a query embedding — it's a pure metadata lookup.

    This is the key to idempotency: if we crashed halfway through a previous run
    we can pick up exactly where we left off without re-indexing anything.
    """
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="chapter",
                    match=MatchValue(value=chapter_name)
                )
            ]
        ),
        limit=1,
        with_payload=False,   # we only need to know if any point exists
        with_vectors=False,
    )
    return len(results) > 0


# ── Step 3: Embed + Index with Full Metadata ──────────────────────────────────

def index_chunks_with_metadata(chunks_with_meta: list, client: QdrantClient, collection_name: str) -> None:
    """
    Embed each chunk and upsert into Qdrant with the full metadata payload.

    Unlike ingestion.ingest.index_chunks (which reads SUBJECT/CHAPTER globals),
    this function accepts metadata already attached to each chunk dict.
    That's essential here because every chapter has different subject/chapter/class_level.

    Uses uuid4 as point IDs so multiple subjects can share the same chunk_index
    values without collision across collections.
    """
    total = len(chunks_with_meta)
    print(f"[{ts()}]   🔢 Embedding {total} chunks...")

    points = []
    for i, item in enumerate(chunks_with_meta):
        response = ollama.embeddings(model=EMBED_MODEL, prompt=item["text"])
        embedding = response["embedding"]

        point = PointStruct(
            id=str(uuid.uuid4()),   # globally unique — avoids ID collisions across runs
            vector=embedding,
            payload=item            # full metadata dict is the payload
        )
        points.append(point)

        if (i + 1) % 10 == 0:
            print(f"[{ts()}]      ... {i + 1}/{total}")

    client.upsert(collection_name=collection_name, points=points)
    print(f"[{ts()}]   ✓  Indexed {total} chunks into '{collection_name}'")


# ── Step 4: Ingest One Chapter ────────────────────────────────────────────────

def ingest_chapter(entry: dict, subject: str, client: QdrantClient) -> dict:
    """
    Full pipeline for a single NCERT chapter: download → parse → chunk → index.

    Returns a result dict with status ('success' | 'skipped' | 'failed'),
    chunk count, and any error message.  The caller collects these to build
    the summary log without crashing the whole batch on one failure.
    """
    chapter      = entry["chapter"]
    class_level  = entry["class_level"]
    collection   = COLLECTION_MAP[subject]
    subject_dir  = subject.lower()
    save_path    = os.path.join("data", subject_dir, entry["filename"])

    # ── 1. Skip if already in Qdrant (idempotent re-run support) ─────────────
    if is_chapter_indexed(client, collection, chapter):
        print(f"[{ts()}]   ↩  Already indexed: {chapter}")
        return {"status": "skipped", "chunks": 0, "chapter": chapter, "subject": subject}

    # ── 2. Download PDF ───────────────────────────────────────────────────────
    ok = download_pdf(entry["url"], save_path)
    if not ok:
        return {
            "status": "failed",
            "chunks": 0,
            "chapter": chapter,
            "subject": subject,
            "error": f"Download failed after {RETRY_COUNT} retries: {entry['url']}"
        }

    # ── 3. Parse + Chunk ──────────────────────────────────────────────────────
    try:
        pages  = parse_pdf(save_path)
        chunks = chunk_text(pages)
    except Exception as e:
        return {"status": "failed", "chunks": 0, "chapter": chapter, "subject": subject, "error": str(e)}

    # ── 4. Attach full metadata to each chunk ─────────────────────────────────
    # We build the payload here (not inside index_chunks_with_metadata) so the
    # chunk_id can encode subject+chapter+position — useful for deduplication
    # and debugging in the Qdrant dashboard.
    chunks_with_meta = []
    for i, chunk in enumerate(chunks):
        chunks_with_meta.append({
            "text":        chunk["text"],
            "subject":     subject,
            "chapter":     chapter,
            "class_level": class_level,
            "source":      f"NCERT {class_level}",
            "page":        chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "chunk_id":    f"{subject}_{chapter}_{i}".replace(" ", "_").lower(),
        })

    # ── 5. Embed and store ────────────────────────────────────────────────────
    try:
        index_chunks_with_metadata(chunks_with_meta, client, collection)
    except Exception as e:
        return {"status": "failed", "chunks": 0, "chapter": chapter, "subject": subject, "error": str(e)}

    return {"status": "success", "chunks": len(chunks_with_meta), "chapter": chapter, "subject": subject}


# ── Step 5: Ensure Collections Exist ─────────────────────────────────────────

def ensure_collections(client: QdrantClient) -> None:
    """
    Create the three subject collections if they don't already exist.

    Called once at the start of every batch run.  Safe to call on an
    already-populated database — create_collection is skipped if exists.
    """
    for subject, collection_name in COLLECTION_MAP.items():
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            print(f"[{ts()}] ✓ Created collection: {collection_name}")
        else:
            print(f"[{ts()}] ✓ Collection exists:  {collection_name}")


# ── Step 6: Batch Runner ──────────────────────────────────────────────────────

def run_batch(subjects: list = None, test_mode: bool = False) -> None:
    """
    Main batch loop.  Downloads and indexes the requested subjects.

    subjects   — list of subject names to process (None = all three)
    test_mode  — if True, only process the first 2 Physics chapters so you
                 can verify the pipeline end-to-end without a 2-hour wait.

    Progress is printed with timestamps.  Failures are logged but don't abort
    the batch — partial success is better than total failure on a slow network.
    """
    run_start = datetime.now()
    subjects  = subjects or list(CORPUS.keys())

    # Build the flat list of (entry, subject) pairs we'll iterate over.
    work_items = []
    for subject in subjects:
        for entry in CORPUS[subject]:
            work_items.append((entry, subject))

    # In test mode, limit to the first 2 Physics chapters only.
    if test_mode:
        work_items = [(e, s) for e, s in work_items if s == "Physics"][:2]
        print(f"[{ts()}] ⚠️  TEST MODE — processing {len(work_items)} chapters only\n")

    total   = len(work_items)
    results = []

    print(f"[{ts()}] ═══════════════════════════════════════")
    print(f"[{ts()}] JEE RAG — Batch Corpus Ingestion")
    print(f"[{ts()}] Subjects : {', '.join(subjects)}")
    print(f"[{ts()}] Chapters : {total}")
    print(f"[{ts()}] ═══════════════════════════════════════\n")

    # Connect once and reuse — opening a new connection per chapter is wasteful.
    client = QdrantClient(host="localhost", port=6333)
    ensure_collections(client)
    print()

    for idx, (entry, subject) in enumerate(work_items, start=1):
        chapter = entry["chapter"]
        print(f"[{ts()}] Processing {idx}/{total}: {subject} — {chapter}")

        try:
            result = ingest_chapter(entry, subject, client)
        except Exception as e:
            # Catch-all so one bad chapter never kills the rest of the batch.
            result = {
                "status":  "failed",
                "chunks":  0,
                "chapter": chapter,
                "subject": subject,
                "error":   str(e),
            }
            print(f"[{ts()}]   ✗  Unexpected error: {e}")

        results.append(result)

        status_icon = {"success": "✅", "skipped": "↩ ", "failed": "❌"}.get(result["status"], "?")
        print(f"[{ts()}]   {status_icon} {result['status'].upper()} — {result.get('chunks', 0)} chunks\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    succeeded     = sum(1 for r in results if r["status"] == "success")
    skipped       = sum(1 for r in results if r["status"] == "skipped")
    failed        = sum(1 for r in results if r["status"] == "failed")
    total_chunks  = sum(r.get("chunks", 0) for r in results)
    elapsed       = round((datetime.now() - run_start).total_seconds(), 1)

    print(f"[{ts()}] ═══════════════════════════════════════")
    print(f"[{ts()}] DONE in {elapsed}s")
    print(f"[{ts()}]   ✅ Succeeded : {succeeded}")
    print(f"[{ts()}]   ↩  Skipped   : {skipped}")
    print(f"[{ts()}]   ❌ Failed    : {failed}")
    print(f"[{ts()}]   📦 Chunks    : {total_chunks}")
    print(f"[{ts()}] ═══════════════════════════════════════")

    # ── Save log ──────────────────────────────────────────────────────────────
    log = {
        "run_timestamp":    run_start.isoformat(),
        "subjects":         subjects,
        "test_mode":        test_mode,
        "total_chapters":   total,
        "succeeded":        succeeded,
        "skipped":          skipped,
        "failed":           failed,
        "total_chunks_added": total_chunks,
        "elapsed_seconds":  elapsed,
        "results":          results,
    }
    os.makedirs("data", exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{ts()}] 📝 Log saved to {LOG_PATH}")

    if failed:
        print(f"\n[{ts()}] Failed chapters:")
        for r in results:
            if r["status"] == "failed":
                print(f"  • {r['subject']} / {r['chapter']} — {r.get('error', 'unknown error')}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """
    Parse CLI arguments and dispatch to run_batch().

    Three modes:
      --test              2 Physics chapters — verify the pipeline works
      --subject Physics   One subject only
      --all               Full corpus (77 chapters, takes ~1-2 hours)
    """
    parser = argparse.ArgumentParser(
        description="Batch ingest NCERT corpus into Qdrant for JEE RAG"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only first 2 Physics chapters"
    )
    group.add_argument(
        "--subject",
        choices=list(CORPUS.keys()),
        help="Ingest a single subject (Physics | Chemistry | Maths)"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest the full NCERT corpus (all subjects)"
    )

    args = parser.parse_args()

    if args.test:
        run_batch(test_mode=True)
    elif args.subject:
        run_batch(subjects=[args.subject])
    elif args.all:
        run_batch()


if __name__ == "__main__":
    main()
