from langchain_core.tools import tool
import boto3
import json
from sentence_transformers import CrossEncoder
from typing import List, Dict
from pinecone import Pinecone
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv(override=True)
# Initialize services
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
    aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)
encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "nvidia-reports"))

def get_content_from_s3(json_source: str) -> Dict:
    """Retrieve content from S3 bucket"""
    bucket = json_source.split('/')[2]
    key = '/'.join(json_source.split('/')[3:])
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = json.loads(response['Body'].read().decode('utf-8'))
        return content
    except Exception as e:
        print(f"Error retrieving from S3: {e}")
        return {}

def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank results using cross-encoder"""
    if not results:
        return []
    
    # Prepare pairs for reranking
    pairs = [(query, result['metadata']['text_preview']) for result in results]
    
    # Get scores from cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Combine results with new scores
    for result, score in zip(results, scores):
        result['score'] = float(score)
    
    # Sort by new scores and return top_k
    reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    return reranked_results

def format_rag_contexts(matches: List[Dict]) -> str:
    contexts = []
    for x in matches:
        # Get full content from S3 if available
        if 'json_source' in x['metadata']:
            full_content = get_content_from_s3(x['metadata']['json_source'])
            chunk_content = full_content.get(x['metadata']['chunk_id'], '')
        else:
            chunk_content = x['metadata'].get('text_preview', '')

        text = (
            f"File Name: {x['metadata']['file_name']}\n"
            f"Year: {x['metadata']['year']}\n"
            f"Quarter: {x['metadata']['quarter']}\n"
            f"Content: {chunk_content}\n"
            f"Source: {x['metadata']['source']}\n"
            f"Relevance Score: {x['score']:.3f}\n"
        )
        contexts.append(text)
    return "\n---\n".join(contexts)

@tool("search_all_namespaces")
def search_all_namespaces(query: str, alpha: float = 0.5):
    """
    Searches across all quarterly report namespaces using hybrid search.
    """
    print(f"\nSearching for query: {query}")
    results = []
    xq = encoder.encode([query])[0].tolist()
    
    namespaces = [f"{year}q{quarter}" for year in range(2023, 2025) 
                 for quarter in range(1, 5)]
    
    print(f"Searching across namespaces: {namespaces}")
    
    for namespace in namespaces:
        try:
            print(f"\nQuerying namespace: {namespace}")
            xc = index.query(
                vector=xq,
                top_k=5,
                include_metadata=True,
                namespace=namespace,
                alpha=alpha,
            )
            if xc["matches"]:
                print(f"Found {len(xc['matches'])} matches in {namespace}")
                results.extend(xc["matches"])
            else:
                print(f"No matches found in {namespace}")
        except Exception as e:
            print(f"Error in namespace {namespace}: {str(e)}")
            continue
    
    print(f"\nTotal results found: {len(results)}")
    
    if results:
        results = rerank_results(query, results)
        return format_rag_contexts(results)
    else:
        return "No results found across any namespace."
    
@tool("search_specific_quarter")
def search_specific_quarter(query: str, year: str, quarter: str, alpha: float = 0.5):
    """
    Searches in a specific quarterly report namespace using hybrid search.
    Args:
        query: The search query
        year: Year of the report (e.g., '2023')
        quarter: Quarter number (1-4)
        alpha: Hybrid search parameter (0 = sparse, 1 = dense)
    """
    namespace = f"{year}q{quarter}"
    xq = encoder.encode([query])[0].tolist()
    
    try:
        # Hybrid search
        xc = index.query(
            vector=xq,
            top_k=10,
            include_metadata=True,
            namespace=namespace,
            alpha=alpha,  # Hybrid search parameter
            
        )
        
        # Rerank results
        if xc["matches"]:
            results = rerank_results(query, xc["matches"])
            return format_rag_contexts(results)
        return "No results found."
    
    except Exception as e:
        return f"Error searching namespace {namespace}: {str(e)}"
    
# ...existing code...

if __name__ == "__main__":
 
    # Test search_all_namespaces
    # query = "NVIDIA financial performance"
    # result = search_all_namespaces.invoke(query)
    # if result:
    #     print("\n=== Search Results ===")
    #     print(result)
    # else:
    #     print("No results found")
        
    # Uncomment to test specific quarter search
    test_query = "NVIDIA revenue growth"
    test_year = "2024"
    test_quarter = "1"
    # Fix: Change the invocation format
    specific_result = search_specific_quarter.invoke({
        "query": test_query,
        "year": test_year,
        "quarter": test_quarter
    })
    print("\n=== Specific Quarter Results ===")
    print(specific_result)
        