# agents/snowflake_agent.py
import os
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv()

def query_snowflake(query: str) -> pd.DataFrame:
    """Execute query against Snowflake."""
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

def get_valuation_summary() -> dict:
    """Get NVIDIA valuation metrics from Snowflake."""
    query = "SELECT * FROM NVDA_VALUATION"
    df = query_snowflake(query)
    
    # Generate text summary
    summary = df.to_string(index=False)
    
    # Generate chart
    plt.figure(figsize=(8, 4))
    plt.bar(df['metric'], df['value'])
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("NVIDIA Valuation Metrics")
    plt.tight_layout()
    
    # Convert chart to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)
    chart_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return {"summary": summary, "chart": chart_base64}

# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)