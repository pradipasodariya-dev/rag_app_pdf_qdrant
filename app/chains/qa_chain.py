from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import BaseRetriever

def create_qa_chain(retriever: BaseRetriever):
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Create RetrievalQA chain using the LLM and retriever
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        # Log and raise any errors during chain creation
        print("Error in create_qa_chain:", e)
        raise
