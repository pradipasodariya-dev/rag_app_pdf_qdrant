from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

def create_qdrant_vector_store(docs, embedding_func, collection_name="pdf_rag"):
    # Get Qdrant URL (default to localhost)
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url)

    # Create collection if it doesn't exist
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    # Create Qdrant vector store from documents
    vector_store = Qdrant.from_documents(
        documents=docs,
        embedding=embedding_func,
        url=qdrant_url,
        collection_name=collection_name
    )

    return vector_store
