# utils/pinecone_utils.py
import os
from pinecone import Pinecone

def get_index(index_name: str, region: str = "us-east-1"):
    """
    Initializes and returns a Pinecone index.
    Expects the environment variable PINECONE_API_KEY to be set.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=pinecone_api_key)
    # You can optionally check for the index existence here if needed
    return pc.Index(index_name)
