# NVIDIA Multi-Agent Research Assistant

This repository implements a multi-agent research assistant integrating:

- **RAG Agent:** Uses Pinecone (index: `nvidia-reports`) with metadata filtering (namespaces like `2023q2`, `2024q1`, etc.) to retrieve historical NVIDIA quarterly reports.
- **Web Search Agent:** Uses SerpAPI for real-time web search related to NVIDIA.
- **Snowflake Agent:** Queries structured NVIDIA valuation measures stored in Snowflake and generates summary charts.

## Setup

1. **Environment Variables:**  
   Create a `.env` file in the project root with:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
   - `SERP_API_KEY`
   - `SNOWFLAKE_USER`
   - `SNOWFLAKE_PASSWORD`
   - `SNOWFLAKE_ACCOUNT`
   - `SNOWFLAKE_WAREHOUSE`
   - `SNOWFLAKE_DATABASE`
   - `SNOWFLAKE_SCHEMA`

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
