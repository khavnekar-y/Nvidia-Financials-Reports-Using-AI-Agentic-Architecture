import os
import sys
from typing import TypedDict, Dict, Any, List, Annotated, Union, Callable
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import operator
from langgraph.graph import StateGraph, END
from graphviz import Digraph
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
# import warnings
# from langchain_core.globals import set_warning_filter
# set_warning_filter("ignore")
# Import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.websearch_agent import search_quarterly
from agents.rag_agent import search_all_namespaces, search_specific_quarter
from agents.snowflake_agent import query_snowflake

class NvidiaGPTState(TypedDict, total=False):
    input: str
    question: str
    year: int
    quarter: int
    web_output: str
    rag_output: str
    snowflake_output: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    final_report: str

# Node Functions
def start_node(state: NvidiaGPTState) -> Dict:
    """Initial node that processes the input"""
    return {"question": state["input"]}

def web_search_node(state: NvidiaGPTState) -> Dict:
    """Execute web search"""
    result = search_quarterly(state["question"])
    return {"web_output": result}

def rag_search_node(state: NvidiaGPTState) -> Dict:
    """Execute RAG search"""
    result = search_all_namespaces(state["question"])
    return {"rag_output": result}

def snowflake_node(state: NvidiaGPTState) -> Dict:
    """Execute Snowflake query"""
    result = query_snowflake(state["question"])
    return {"snowflake_output": result}

@tool("final_report")
def final_report(
    introduction: str,
    key_findings: List[str],
    analysis: str,
    conclusion: str,
    sources: List[str]
) -> Dict:
    """Generates final report in structured format"""
    return {
        "introduction": introduction,
        "key_findings": key_findings,
        "analysis": analysis,
        "conclusion": conclusion,
        "sources": sources
    }

def final_report_node(state: NvidiaGPTState) -> Dict:
    """Generate final report combining all sources"""
    return {
        "final_report": final_report(
            introduction=f"Analysis of query: {state['question']}",
            key_findings=[
                state.get("web_output", "No web data"),
                state.get("rag_output", "No RAG data"),
                state.get("snowflake_output", "No Snowflake data")
            ],
            analysis="Combined analysis of all sources",
            conclusion="Final conclusions based on all available data",
            sources=["Web Search", "RAG Search", "Snowflake Query"]
        )
    }

# Initialize NvidiaGPT
def initialize_nvidia_gpt():
    system_prompt = """You are NvidiaGPT, an AI assistant specialized in NVIDIA financial analysis.
    You have access to multiple data sources and tools. Use them wisely to provide comprehensive analysis."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ])

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [search_quarterly, search_all_namespaces, search_specific_quarter, 
             query_snowflake, final_report]
    
    return prompt.partial(llm=llm).bind_tools(tools, tool_choice="any")

def generate_workflow_diagram(filename="nvidia_workflow.png"):
    """Generates and saves workflow diagram"""
    dot = Digraph(comment='NVIDIA Analysis Pipeline')
    dot.attr(rankdir='LR')
    
    # Add nodes
    dot.node('start', 'Start')
    dot.node('web_search', 'Web Search')
    dot.node('rag_search', 'RAG Search')
    dot.node('snowflake', 'Snowflake')
    dot.node('report_generator', 'Report Generator')
    
    # Add edges
    dot.edge('start', 'web_search')
    dot.edge('start', 'rag_search')
    dot.edge('start', 'snowflake')
    dot.edge('web_search', 'report_generator')
    dot.edge('rag_search', 'report_generator')
    dot.edge('snowflake', 'report_generator')
    
    # Generate diagram
    try:
        dot.render(filename, format='png', cleanup=True)
        return f"{filename}.png"
    except Exception as e:
        print(f"Warning: Could not generate diagram: {e}")
        return None

def build_pipeline():
    """Build and return the compiled pipeline"""
    graph = StateGraph(NvidiaGPTState)
    
    # Add nodes with proper callable functions
    graph.add_node("start", start_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("snowflake", snowflake_node)
    graph.add_node("report_generator", final_report_node)  # Changed from "final_report"
    
    # Set flow
    graph.set_entry_point("start")
    graph.add_edge("start", "web_search")
    graph.add_edge("start", "rag_search")
    graph.add_edge("start", "snowflake")
    graph.add_edge("web_search", "report_generator")  # Changed
    graph.add_edge("rag_search", "report_generator")  # Changed
    graph.add_edge("snowflake", "report_generator")  # Changed
    graph.add_edge("report_generator", END)  # Changed
    
    return graph.compile()

if __name__ == "__main__":
    try:
        # Generate workflow diagram (one-time)
        # diagram_path = generate_workflow_diagram()
        # if diagram_path:
        #     print(f"Workflow diagram saved to: {diagram_path}")
        
        # Initialize and run pipeline
        pipeline = build_pipeline()
        result = pipeline.invoke({
            "input": "Analyze NVIDIA's financial performance in Q4 2023",
            "chat_history": [],
            "intermediate_steps": []
        })
        
        print("\nAnalysis Results:")
        print(result.get("final_report"))
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())