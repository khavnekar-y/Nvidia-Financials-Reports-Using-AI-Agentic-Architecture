# agents/snowflake_agent.py
import os
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import time
from typing import Dict, Any

load_dotenv(override=True)

# Initialize Snowflake connection
conn = snowflake.connector.connect(
    user=os.environ.get('SNOWFLAKE_USER'),
    password=os.environ.get('SNOWFLAKE_PASSWORD'),
    account=os.environ.get('SNOWFLAKE_ACCOUNT'),
    warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE'),
    database=os.environ.get('SNOWFLAKE_DATABASE'),
    schema=os.environ.get('SNOWFLAKE_SCHEMA')
)
 
def query_snowflake(question: str) -> Dict:
    """Query Snowflake with predefined queries based on question intent"""
    # Default query for financial metrics
    base_query = "SELECT * FROM Valuation_Measures ORDER BY DATE DESC LIMIT 5"
    
    try:
        # Execute query and get DataFrame
        df = pd.read_sql(base_query, conn)
        
        # Generate summary
        summary = {
            "metrics": df.to_dict('records'),
            "latest_date": str(df['DATE'].max()),
            "query_status": "success"
        }
        
        return summary
        
    except Exception as e:
        return {
            "error": str(e),
            "query_status": "failed"
        }

 
def get_valuation_summary(query:str=None) -> dict:
    """Get NVIDIA valuation metrics visualization"""
    try:
        # Use base query
        df = pd.read_sql("SELECT * FROM Valuation_Measures ORDER BY DATE DESC LIMIT 5", conn)
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        for date in df["DATE"].unique():
            subset = df[df["DATE"] == date]
            plt.bar(subset.columns[1:], subset.iloc[0, 1:], label=str(date))
        
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("NVIDIA Valuation Metrics")
        plt.xticks(rotation=45)
        plt.legend()
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "chart": img_str,
            "summary": df.to_string(),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }
         
# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  
    temperature=0,
    anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')  # Get from environment instead of hardcoding
) 

try:
    # Create agent with the tool
    # Simplify agent initialization
    agent = initialize_agent(
        tools=[snowflake_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Add specific agent type
        handle_parsing_errors=True,
        max_iterations=2,  # Limit iterations to reduce token usage
        early_stopping_method="generate"  # Add early stopping
    )
except Exception as e:
    print(f"Error initializing agent: {str(e)}")
    print("Available Claude models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307")
    raise
 
def get_ai_analysis():
    """Get AI-generated analysis of NVIDIA metrics"""
    prompt = """Analyze NVIDIA financial metrics using the nvidia_financial_metrics tool.
    Provide a brief summary of key insights."""
    
    try:
        response = agent.invoke({"input": prompt})
        return response.get("output", str(response))
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return "Analysis unavailable - Rate limit exceeded. Please try again later."  

if __name__ == "__main__":
    analysis = get_ai_analysis()
    print(analysis)