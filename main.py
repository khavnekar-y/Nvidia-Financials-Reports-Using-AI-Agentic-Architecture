from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from langGraph.pipeline import build_pipeline

app = FastAPI()
graph = build_pipeline()

class QueryRequest(BaseModel):
    question: str
    search_type: str
    selected_periods: List[str]

@app.post("/research_report")
async def research_report(request: QueryRequest):
    try:
        # Create message for current request
        current_message = {
            "role": "user",
            "content": request.question,
            "search_type": request.search_type,
            "selected_periods": request.selected_periods
        }
        
        # Use empty chat history for now (stateless)
        chat_history = [current_message]
        
        # Prepare state
        state = {
            "input": request.question,
            "question": request.question,
            "search_type": request.search_type,
            "selected_periods": request.selected_periods,
            "chat_history": chat_history,
            "intermediate_steps": []
        }
        
        # Execute pipeline
        result = graph.invoke(state)
        
        # Check for Snowflake data availability
        include_snowflake = (
            request.search_type == "Specific Quarter" and
            any(p.startswith(("2024", "2025")) for p in request.selected_periods)
        )
        
        # Format response with structured report sections
        if isinstance(result.get("final_report"), dict):
            final_report = format_report(result["final_report"])
        else:
            final_report = str(result.get("final_report", "No report available"))
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": final_report,
            "rag_output": result.get("rag_output", {}),
            "snowflake_data": result.get("valuation_data", {}) if include_snowflake else {}
        }
        chat_history.append(assistant_message)
        
        # Prepare response
        response = {
            "final_report": final_report,
            "rag_output": result.get("rag_output", {}),
            "valuation_data": result.get("valuation_data", {}) if include_snowflake else {},
            "chat_history": chat_history
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

def format_report(report_dict: Dict) -> str:
    """Format a structured report dictionary into markdown text"""
    sections = [
        f"# {report_dict.get('introduction', 'NVIDIA Analysis')}",
        "## Key Findings",
        "\n".join([f"- {finding}" for finding in report_dict.get('key_findings', [])]),
        f"## Analysis\n{report_dict.get('analysis', '')}",
        f"## Conclusion\n{report_dict.get('conclusion', '')}",
        "## Sources",
        "\n".join([f"- {source}" for source in report_dict.get('sources', [])])
    ]
    return "\n\n".join(sections)    
