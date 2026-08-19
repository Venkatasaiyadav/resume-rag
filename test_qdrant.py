"""Quick test to verify Qdrant is working"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv

load_dotenv()

# Connect
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

# List collections
collections = client.get_collections()
print("✅ Connected to Qdrant!")
print(f"📊 Existing collections: {[c.name for c in collections.collections]}")

# Test collection creation
test_collection = "test_collection"
try:
    client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    print(f"✅ Created test collection")
    
    # Add a test point
    client.upsert(
        collection_name=test_collection,
        points=[
            PointStruct(
                id=1,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"text": "test"}
            )
        ]
    )
    print("✅ Added test point")
    
    # Search
    results = client.search(
        collection_name=test_collection,
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=1
    )
    print(f"✅ Search results: {results}")
    
    # Cleanup
    client.delete_collection(test_collection)
    print("✅ Cleaned up test collection")
    
except Exception as e:
    print(f"❌ Error: {e}")