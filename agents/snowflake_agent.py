# agents/snowflake_agent.py
import os
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import base64

load_dotenv()

def query_snowflake(query: str) -> pd.DataFrame:
    """
    Connects to Snowflake and executes a SQL query.
    Requires environment variables: SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT, 
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA.
    """
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
    """
    Queries Snowflake for NVIDIA valuation measures.
    Assumes a table named "NVDA_VALUATION" exists.
    Returns a dictionary containing a textual summary and a base64‑encoded bar chart image.
    """
    query = "SELECT * FROM NVDA_VALUATION"  # adjust as needed
    df = query_snowflake(query)
    
    summary = df.to_string(index=False)
    
    # Generate a bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(df['metric'], df['value'])
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("NVIDIA Valuation Metrics")
    plt.tight_layout()
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close()
    img_bytes.seek(0)
    chart_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return {"summary": summary, "chart": chart_base64}
