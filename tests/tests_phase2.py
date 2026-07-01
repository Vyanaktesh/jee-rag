# tests/test_phase2.py

from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check total count
count = client.count(collection_name="jee_chunks")
print(f"Total chunks: {count.count}")

# Fetch first 3 chunks and inspect them
results = client.scroll(
    collection_name="jee_chunks",
    limit=3,
    with_payload=True,
    with_vectors=False   # don't print the 768 numbers, too noisy
)

print("\n--- Sample Chunks ---")
for point in results[0]:
    print(f"\nChunk ID    : {point.payload['chunk_id']}")
    print(f"Page        : {point.payload['page']}")
    print(f"Subject     : {point.payload['subject']}")
    print(f"Chapter     : {point.payload['chapter']}")
    print(f"Text preview: {point.payload['text'][:200]}...")
    print("-" * 60)