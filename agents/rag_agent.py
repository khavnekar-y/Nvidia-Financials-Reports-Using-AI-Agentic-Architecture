# agents/rag_agent.py
import os
from langchain_community.vectorstores import Pinecone 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv

load_dotenv()

def query_nvidia_reports(query: str, year: int = None, quarter: int = None, top_k: int = 5) -> str:
    """Query NVIDIA reports using vector search."""
    # Create namespace from year and quarter
    namespace = f"{year}q{quarter}" if year and quarter else None
    
    # Initialize Pinecone client and vector store
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "nvidia-reports"
    index = pc.Index(index_name)
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store with the embeddings model
    vector_store = Pinecone(index, embeddings, text_key="page_content")
    
    # Create a retriever with namespace filtering if specified
    search_kwargs = {"k": top_k}
    if namespace:
        search_kwargs["namespace"] = namespace
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # Create a QA chain with the retriever
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    
    # Run the query
    result = qa_chain.invoke(query)
    return result["result"]

# Create LangChain tool for the RAG agent
from langchain.tools import Tool

rag_tool = Tool(
    name="nvidia_reports_search",
    description="Search NVIDIA quarterly reports for specific information",
    func=query_nvidia_reports
)