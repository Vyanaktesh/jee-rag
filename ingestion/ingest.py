# ingestion/ingest.py

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
import uuid
import re

# ── Configuration ────────────────────────────────────────────────
PDF_PATH     = "data/laws_of_motion.pdf"
SUBJECT      = "Physics"
CHAPTER      = "Work Energy and Power"
SOURCE       = "NCERT Class 11"
COLLECTION   = "jee_chunks"
CHUNK_SIZE   = 500      # characters per chunk (we use chars not tokens for simplicity)
OVERLAP      = 100      # overlap between chunks
EMBED_MODEL  = "nomic-embed-text"

# ── Step 1: Parse PDF ────────────────────────────────────────────
def parse_pdf(pdf_path):
    """
    Opens the PDF and extracts text page by page.
    Returns a list of (page_number, text) tuples.
    """
    print(f"\n📄 Parsing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Clean up the text — remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)      # multiple newlines → single
        text = re.sub(r' +', ' ', text)         # multiple spaces → single
        text = text.strip()

        if text:  # skip empty pages
            pages.append((page_num + 1, text))  # page_num is 0-indexed, so +1

    print(f"   ✓ Extracted {len(pages)} pages")
    return pages


# ── Step 2: Chunk Text ───────────────────────────────────────────
def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Takes pages and splits them into overlapping chunks.
    Each chunk remembers which page it came from.
    """
    print(f"\n✂️  Chunking text (size={chunk_size}, overlap={overlap})")
    chunks = []

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Get the chunk
            chunk = text[start:end]

            # Don't create tiny leftover chunks (less than 100 chars)
            if len(chunk) < 100:
                break

            chunks.append({
                "text":        chunk,
                "page":        page_num,
                "chunk_index": len(chunks)
            })

            # Move forward by (chunk_size - overlap)
            # This creates the sliding window effect
            start += (chunk_size - overlap)

    print(f"   ✓ Created {len(chunks)} chunks")
    return chunks


# ── Step 3: Embed + Index ────────────────────────────────────────
def index_chunks(chunks, client, collection_name):
    """
    Takes each chunk, creates an embedding, attaches metadata,
    and stores everything in Qdrant.
    """
    print(f"\n🔢 Embedding and indexing {len(chunks)} chunks...")

    points = []
    for i, chunk in enumerate(chunks):

        # Create embedding for this chunk
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=chunk["text"]
        )
        embedding = response["embedding"]

        # Build the Qdrant point — vector + metadata
        point = PointStruct(
            id=i,                          # unique id
            vector=embedding,              # 768-dim vector
            payload={
                "text":        chunk["text"],
                "subject":     SUBJECT,
                "chapter":     CHAPTER,
                "source":      SOURCE,
                "page":        chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "chunk_id":    f"{SUBJECT}_{CHAPTER}_{i}".replace(" ", "_").lower()
            }
        )
        points.append(point)

        # Progress indicator every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"   ... embedded {i + 1}/{len(chunks)} chunks")

    # Bulk upload all points to Qdrant in one call
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"   ✓ Indexed all {len(chunks)} chunks into Qdrant")


# ── Main Pipeline ────────────────────────────────────────────────
def main():
    print("=== JEE RAG Ingestion Pipeline ===")

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"\n✓ Created collection: {COLLECTION}")
    else:
        print(f"\n✓ Collection exists: {COLLECTION}")

    # Run the three steps
    pages  = parse_pdf(PDF_PATH)
    chunks = chunk_text(pages)
    index_chunks(chunks, client, COLLECTION)

    # Verify — check how many points are now in Qdrant
    count = client.count(collection_name=COLLECTION)
    print(f"\n✅ Done! Qdrant now has {count.count} chunks indexed.")
    print(f"   Collection: {COLLECTION}")
    print(f"   Subject:    {SUBJECT}")
    print(f"   Chapter:    {CHAPTER}")


if __name__ == "__main__":
    main()