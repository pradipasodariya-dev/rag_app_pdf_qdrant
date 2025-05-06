from app.loaders.pdf_loader import load_and_split_pdf
from app.embeddings.embedder import get_embedding_function
from app.retriever.qdrant_store import create_qdrant_vector_store
from app.chains.qa_chain import create_qa_chain

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Path to PDF
    file_path = "data/ScrumGuide.pdf"  
    # Example user query
    query = "What is the scrum framework?"  

    print("Loading and splitting document...")
    # Load and chunk PDF
    docs = load_and_split_pdf(file_path) 

    print("Initializing embedding function...")
    # Load embedding model
    embedding_func = get_embedding_function()  

    print("Creating Qdrant vector store...")
    # Store embeddings in Qdrant
    vector_store = create_qdrant_vector_store(docs, embedding_func)  

    print("Creating QA chain...")
    # Set up retriever with top-5 results
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # Build retrieval QA chain  
    qa_chain = create_qa_chain(retriever)  

    print("Querying...")
    # Run query through the chain
    answer = qa_chain.run(query)  
    print(f"\n Answer:\n{answer}")

# Run the main function
if __name__ == "__main__":
    main()
