from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import ollama

print("=== Phase 1 Connection Test ===\n")

# 1. Ollama embedding
print("1. Testing Ollama embeddings...")
response = ollama.embeddings(
    model="nomic-embed-text",
    prompt="Newton's second law states F = ma"
)
embedding = response["embedding"]
print(f"   ✓ Got embedding — dimension: {len(embedding)}")

# 2. Qdrant connection
print("\n2. Testing Qdrant connection...")
client = QdrantClient(host="localhost", port=6333)
collections = client.get_collections()
print(f"   ✓ Connected — existing collections: {collections}")

# 3. Create a test collection (updated — no deprecation warning)
print("\n3. Creating test collection...")
if client.collection_exists("test_phase1"):
    client.delete_collection("test_phase1")
client.create_collection(
    collection_name="test_phase1",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("   ✓ Collection created")

# 4. Insert a test chunk with metadata
print("\n4. Inserting test chunk...")
client.upsert(
    collection_name="test_phase1",
    points=[
        PointStruct(
            id=1,
            vector=embedding,
            payload={
                "text": "Newton's second law states F = ma",
                "subject": "Physics",
                "chapter": "Laws of Motion",
                "source": "NCERT Class 11",
                "page": 92
            }
        )
    ]
)
print("   ✓ Chunk inserted with metadata")

# 5. Retrieve it back (updated — query_points instead of search)
print("\n5. Querying for similar content...")
query_embedding = ollama.embeddings(
    model="nomic-embed-text",
    prompt="What is force equal to?"
)["embedding"]

results = client.query_points(
    collection_name="test_phase1",
    query=query_embedding,
    limit=1
).points

print(f"   ✓ Retrieved: '{results[0].payload['text']}'")
print(f"   ✓ Score: {results[0].score:.4f}")
print(f"   ✓ Source: {results[0].payload['source']}, page {results[0].payload['page']}")

# 6. Metadata filter test
print("\n6. Testing metadata filter (Physics only)...")
results_filtered = client.query_points(
    collection_name="test_phase1",
    query=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="subject", match=MatchValue(value="Physics"))]
    ),
    limit=1
).points
print(f"   ✓ Filtered retrieval works — got {len(results_filtered)} result(s)")

print("\n=== All checks passed. Stack is ready. ===")