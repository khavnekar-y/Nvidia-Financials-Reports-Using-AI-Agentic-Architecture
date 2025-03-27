# agents/snowflake_agent.py
import os
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI  # Updated import
from dotenv import load_dotenv
import time
 
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
 
def get_valuation_summary(query: str = None) -> dict:
    """Get NVIDIA valuation metrics from Snowflake."""
    query = "SELECT * FROM Valuation_Measures"
    df = query_snowflake(query)
   
    # Transform the DataFrame into a long format
    df_long = df.melt(id_vars=["DATE"], var_name="metric", value_name="value")
   
    # Truncate the DataFrame to reduce the size of the summary
    df_long = df_long.head(5)  # Limit to the first 5 rows for the summary
   
    # Generate text summary
    summary = df_long.to_string(index=False)
    print(f"Summary length: {len(summary)} characters")  # Debugging
    print(f"Summary content:\n{summary}")  # Debugging
   
    # Generate chart
    plt.figure(figsize=(10, 6))
    for date in df["DATE"].unique():
        subset = df_long[df_long["DATE"] == date]
        plt.bar(subset["metric"], subset["value"], label=str(date))
   
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("NVIDIA Valuation Metrics Over Time")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Date")
    plt.tight_layout()
   
    # Save chart as a .png file
    chart_file_path = "nvidia_valuation_metrics.png"
    plt.savefig(chart_file_path, format="png")
    plt.close()
   
    # Convert chart to base64
    with open(chart_file_path, "rb") as img_file:
        chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')
   
    return {"summary": summary, "chart": chart_base64}
 
# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)
 
# Initialize OpenAI model with ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Use "gpt-4" if available
 
# Create agent with the tool
agent = initialize_agent(
    tools=[snowflake_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
 
def get_ai_analysis():
    """Get AI-generated analysis of NVIDIA metrics"""
    prompt = """You are an AI agent tasked with analyzing NVIDIA financial metrics.
    Use the nvidia_financial_metrics tool to get the data and provide a summary of key insights.
    Your response must follow this format:
   
    Thought: [Your thought process]
    Action: [The action you will take, e.g., "nvidia_financial_metrics"]
    Action Input: [The input for the action, if any]
    """
   
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = agent.run(prompt)
            return response
        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower():
                print("Rate limit exceeded. Retrying...")
                time.sleep(10)  # Wait for 10 seconds before retrying
            else:
                print(f"Error during analysis: {error_message}")
                return None
    print("Failed after 3 attempts due to rate limit.")
    return None
 

if __name__ == "__main__":
    analysis = get_ai_analysis()
    print(analysis)