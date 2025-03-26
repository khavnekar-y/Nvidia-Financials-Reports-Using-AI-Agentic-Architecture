# langgraph_pipeline/pipeline.py
import os
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Dict, Any
from agents.rag_agent import query_nvidia_reports
from agents.web_search_agent import search_quarterly
from agents.snowflake_agent import get_valuation_summary

class MultiAgentState(TypedDict, total=False):
    question: str
    year: int
    quarter: int
    rag_output: str
    web_output: dict
    snowflake_output: dict
    final_report: str

def combined_agent(state: MultiAgentState) -> Dict[str, Any]:
    """
    Combines responses from the RAG Agent, Web Search Agent, and Snowflake Agent
    to produce a comprehensive research report.
    """
    question = state.get("question", "Summarize NVIDIA's performance.")
    year = state.get("year", 2023)
    quarter = state.get("quarter", 1)
    
    # RAG Agent: Query Pinecone with metadata filtering
    rag_result = query_nvidia_reports(question, year, quarter)
    
    # Web Search Agent: Perform real-time web search
    web_result = search_quarterly(year, quarter)
    
    # Snowflake Agent: Retrieve structured valuation measures
    snowflake_result = get_valuation_summary()
    
    # Combine results into a final report
    report = f"""
RESEARCH REPORT on NVIDIA (Q{quarter} {year})
===============================================

HISTORICAL PERFORMANCE (RAG Agent):
-----------------------------------
{rag_result}

REAL-TIME INDUSTRY INSIGHTS (Web Search Agent):
-----------------------------------------------
{web_result}

STRUCTURED FINANCIAL VALUATION (Snowflake Agent):
-------------------------------------------------
{snowflake_result['summary']}

[Valuation Chart (base64 encoded PNG):]
{snowflake_result['chart']}
    """
    return {"final_report": report}

def build_graph():
    builder = StateGraph(MultiAgentState)
    builder.add_node("CombinedAgent", RunnableLambda(combined_agent))
    builder.set_entry_point("CombinedAgent")
    builder.add_edge("CombinedAgent", END)
    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    sample_state = {
        "question": "What are the key factors affecting NVIDIA's performance?",
        "year": 2023,
        "quarter": 2
    }
    result = graph.invoke(sample_state)
    print("\n📝 Final Research Report:\n")
    print(result.get("final_report"))
