from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    # Create and return an embedding function using a pretrained Hugging Face model
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
