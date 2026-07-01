# retrieval/retriever.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
import ollama

# ── Collection routing ────────────────────────────────────────────
# Three separate collections — one per subject — so retrieval only
# searches relevant content.  A Physics query never touches Maths chunks.
COLLECTION_MAP = {
    "Physics":   "physics_chunks",
    "Chemistry": "chemistry_chunks",
    "Maths":     "maths_chunks",
}
DEFAULT_COLLECTION = "physics_chunks"

EMBED_MODEL = "nomic-embed-text"
TOP_K       = 5


# ── Load all chunks from Qdrant into memory for BM25 ─────────────
def load_all_chunks(client, collection_name):
    """
    BM25 needs all documents in memory to build its index.
    We scroll the specified collection only — building a global BM25
    across all subjects would let Maths terms pollute Physics rankings.
    """
    all_chunks = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        all_chunks.extend(results)
        if offset is None:
            break

    print(f"   Loaded {len(all_chunks)} chunks for BM25 index")
    return all_chunks


# ── Build BM25 Index ─────────────────────────────────────────────
def build_bm25_index(chunks):
    """
    BM25Okapi takes a list of tokenized documents.
    Tokenization here is simple — just split by whitespace.
    In production you'd use a proper tokenizer.
    """
    tokenized = [chunk.payload["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25


# ── Dense Retrieval ──────────────────────────────────────────────
def dense_retrieve(query, client, top_k, collection_name, subject_filter=None):
    """
    Embed the query and find nearest vectors in the given collection.

    subject_filter is kept as a parameter for backward compatibility but
    is no longer used as a Qdrant payload filter — the collection itself
    already scopes the search to one subject, so a metadata filter would
    be redundant.  It's preserved here in case we later want chapter-level
    filtering within a subject collection.
    """
    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]

    # No subject filter needed — the collection IS the subject scope.
    # Keeping Filter import available for future chapter-level filtering.
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k * 2   # fetch more than needed; RRF will re-rank
    ).points

    return results


# ── BM25 Retrieval ───────────────────────────────────────────────
def bm25_retrieve(query, chunks, bm25, top_k):
    """
    Tokenize the query and score all chunks using BM25.
    Returns top_k chunks sorted by BM25 score.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k * 2]


# ── Reciprocal Rank Fusion ───────────────────────────────────────
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    """
    Merges two ranked lists using RRF formula:
    score(chunk) = sum of 1/(k + rank) across all lists

    k=60 is the standard constant — prevents top ranks
    from having too much influence.
    """
    rrf_scores = {}

    for rank, point in enumerate(dense_results):
        chunk_id = point.payload["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {"score": 0, "payload": point.payload}
        rrf_scores[chunk_id]["score"] += 1 / (k + rank + 1)

    for rank, (chunk, bm25_score) in enumerate(bm25_results):
        chunk_id = chunk.payload["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {"score": 0, "payload": chunk.payload}
        rrf_scores[chunk_id]["score"] += 1 / (k + rank + 1)

    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    return sorted_results[:5]


# ── Main Hybrid Retriever ────────────────────────────────────────
def hybrid_retrieve(query, subject_filter=None, collection_name=None):
    """
    Full hybrid retrieval pipeline:
    1. Resolve which Qdrant collection to search
    2. Dense retrieval (vector similarity)
    3. BM25 retrieval (keyword matching) from the same collection
    4. RRF fusion of both results

    Collection resolution priority:
      collection_name (explicit) > subject_filter (via COLLECTION_MAP) > DEFAULT_COLLECTION
    """
    # Resolve collection — explicit name wins, then subject lookup, then default.
    if collection_name:
        resolved_collection = collection_name
    elif subject_filter:
        resolved_collection = COLLECTION_MAP.get(subject_filter, DEFAULT_COLLECTION)
    else:
        resolved_collection = DEFAULT_COLLECTION

    client = QdrantClient(host="localhost", port=6333)

    print(f"\n🔍 Query: '{query}'")
    print(f"   Subject    : {subject_filter or 'unknown'}")
    print(f"   Collection : {resolved_collection}")

    # Load chunks from the correct subject collection only.
    # BM25 built from physics_chunks won't contain Maths vocabulary,
    # keeping keyword scores meaningful within the subject domain.
    print("\n   Building BM25 index...")
    all_chunks = load_all_chunks(client, resolved_collection)
    bm25 = build_bm25_index(all_chunks)

    print("   Running dense retrieval...")
    dense_results = dense_retrieve(query, client, TOP_K, resolved_collection)

    print("   Running BM25 retrieval...")
    bm25_results = bm25_retrieve(query, all_chunks, bm25, TOP_K)

    print("   Fusing with RRF...")
    final_results = reciprocal_rank_fusion(dense_results, bm25_results)

    return final_results


# ── Test routing to all three collections ────────────────────────
if __name__ == "__main__":

    print("\n--- Test 1: Physics ---")
    results = hybrid_retrieve("What is work done by a force?", subject_filter="Physics")
    if results:
        print(f"Top result: {results[0][1]['payload']['text'][:150]}")

    print("\n--- Test 2: Chemistry ---")
    results = hybrid_retrieve("What is an ionic bond?", subject_filter="Chemistry")
    if results:
        print(f"Top result: {results[0][1]['payload']['text'][:150]}")

    print("\n--- Test 3: Maths ---")
    results = hybrid_retrieve("What is integration?", subject_filter="Maths")
    if results:
        print(f"Top result: {results[0][1]['payload']['text'][:150]}")
