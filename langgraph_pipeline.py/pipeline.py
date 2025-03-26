# langgraph_pipeline/pipeline.py
import os
import io
import base64
from typing import TypedDict, Dict, Any, List, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.websearch_agent import search_quarterly

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Enhanced state with chat history support
class NvidiaAgentState(TypedDict, total=False):
    # Core input
    input: str
    question: str  # Main research question
    year: int      # Target year
    quarter: int   # Target quarter
    
    # Agent outputs
    web_output: str
    
    # Conversation tracking
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    
    # Final output
    final_report: str

def web_search_agent(state: NvidiaAgentState) -> Dict[str, Any]:
    """Web search agent for real-time information"""
    year = state.get("year", 2023)
    quarter = state.get("quarter", 1)
    
    # Call web search function
    result = search_quarterly(year, quarter)
    
    # Record the action
    action = AgentAction(
        tool="nvidia_web_search", 
        tool_input={"year": year, "quarter": quarter},
        log=f"Searching web for NVIDIA Q{quarter} {year} information"
    )
    
    return {
        "web_output": result,
        "intermediate_steps": [(action, result)],
        "chat_history": state.get("chat_history", []) + [
            AIMessage(content=f"I've gathered real-time web information about NVIDIA's Q{quarter} {year} results.")
        ]
    }

def report_generator(state: NvidiaAgentState) -> Dict[str, Any]:
    """Generates the final report based only on web search results"""
    question = state.get("question", state.get("input", "NVIDIA performance analysis"))
    year = state.get("year", 2023)
    quarter = state.get("quarter", 1)
    
    web_result = state.get("web_output", "No web search results available.")
    
    # Create focused report with just web search results
    report = f"""
NVIDIA WEB SEARCH REPORT (Q{quarter} {year})
============================================

QUESTION: {question}

REAL-TIME INDUSTRY INSIGHTS:
---------------------------
{web_result}
"""
    
    return {
        "final_report": report,
        "chat_history": state.get("chat_history", []) + [
            AIMessage(content="I've compiled web search information into a report.")
        ]
    }
def generate_graph_diagram(graph, filename="nvidia_workflow_diagram.png"):
    """Generate and save a PNG visualization of the graph"""
    try:
        import os
        
        # Get PNG data from compiled graph
        png_data = graph.get_graph().draw_png()
        
        # Save to file
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, "wb") as f:
            f.write(png_data)
            
        print(f"Graph visualization saved to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        return None


def build_graph():
    """Builds and returns the compiled workflow graph with just web search"""
    # Initialize the graph with our state
    builder = StateGraph(NvidiaAgentState)
    
    # Add only the web search agent and report generator
    builder.add_node("WEB_AGENT", RunnableLambda(web_search_agent))
    builder.add_node("REPORT_GENERATOR", RunnableLambda(report_generator))
    
    # Create simplified flow
    builder.set_entry_point("WEB_AGENT")
    builder.add_edge("WEB_AGENT", "REPORT_GENERATOR")
    builder.add_edge("REPORT_GENERATOR", END)
    diagram_path = generate_graph_diagram(builder)
    
    # Compile the graph
    graph = builder.compile()
    
    return graph,diagram_path

if __name__ == "__main__":
    # Build the graph
    graph,diagram_path = build_graph()
    
    # Create initial state with a sample question
    initial_state = {
        "question": "What are the key factors driving NVIDIA's performance?",
        "year": 2023,
        "quarter": 2,
        "chat_history": [
            HumanMessage(content="What are the key factors driving NVIDIA's performance?")
        ],
        "intermediate_steps": []
    }
    
    # Execute the workflow
    result = graph.invoke(initial_state)
    
    print("\n📊 FINAL WEB SEARCH REPORT:\n")
    print(result.get("final_report"))
    
    # Show conversation history
    print("\n💬 CONVERSATION HISTORY:\n")
    for msg in result.get("chat_history", []):
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
    print(f"\nGraph diagram saved to {diagram_path}")