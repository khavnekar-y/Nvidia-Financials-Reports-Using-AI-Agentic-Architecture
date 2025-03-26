# agents/rag_agent.py
import os
import openai
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not found in environment variables")

def query_nvidia_reports(query: str, year: int, quarter: int, top_k: int = 5) -> str:
    """
    Query the Pinecone index "nvidia-reports" from a specific namespace (e.g. "2023q2")
    using hybrid search to generate an answer.
    """
    namespace = f"{year}q{quarter}"
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "nvidia-reports"
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="page_content")
    
    # Use namespace filtering in the retriever:
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k, "namespace": namespace})
    semantic_results = retriever.get_relevant_documents(query)
    keyword_results = vector_store.similarity_search(query, k=top_k, namespace=namespace)
    
    # Merge and deduplicate results (with weighted scores)
    ranked_results = [(doc.page_content, 0.7) for doc in semantic_results] + \
                     [(doc.page_content, 0.3) for doc in keyword_results]
    unique_results = {}
    for content, score in sorted(ranked_results, key=lambda x: x[1], reverse=True):
        unique_results[content] = score
    final_results = list(unique_results.keys())[:top_k]
    
    if not final_results:
        return f"No relevant information found for namespace {namespace}"
    
    top_chunks = "\n\n".join(final_results[:3])
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI financial assistant that answers questions based on NVIDIA quarterly reports."},
            {"role": "user", "content": f"Context:\n{top_chunks}\n\nQuestion: {query}\nAnswer based on the above context:"}
        ]
    )
    return response.choices[0].message.content
