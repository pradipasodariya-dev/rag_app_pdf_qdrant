from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define a function to load and split a PDF document
def load_and_split_pdf(file_path: str):
    # Initialize the PDF loader with the given file path
    loader = PyPDFLoader(file_path)
    
    # Load the document into memory; this returns a list of Document objects
    documents = loader.load()

    # Create a text splitter that breaks text into chunks of 1000 characters
    # with an overlap of 200 characters between chunks to preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        # Maximum size of each chunk
        chunk_size=1000,   
        # Number of characters to overlap between chunks
        chunk_overlap=200  
    )

    # Split the loaded documents into chunks using the text splitter
    return text_splitter.split_documents(documents)
