import os
import sys
from typing import TypedDict, Dict, Any, List, Annotated, Union, Callable
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, AgentExecutor, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import operator
from langgraph.graph import StateGraph, END
from graphviz import Digraph
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.websearch_agent import search_quarterly
from agents.rag_agent import search_all_namespaces, search_specific_quarter
from agents.snowflake_agent import query_snowflake

def create_tools():
    """Create properly formatted tools for the agent"""
    tools = [
        Tool(
            name="web_search",
            func=search_quarterly,
            description="Search for NVIDIA quarterly financial information from web sources"
        ),
        Tool(
            name="rag_search",
            func=search_all_namespaces,
            description="Search across all document repositories for NVIDIA information"
        ),
        Tool(
            name="specific_quarter_search",
            func=search_specific_quarter,
            description="Search for specific quarter information from NVIDIA reports"
        ),
        Tool(
            name="snowflake_query",
            func=query_snowflake,
            description="Query Snowflake database for NVIDIA financial metrics"
        ),
        Tool(
            name="generate_report",
            func=final_report,
            description="Generate a structured report from analyzed information"
        )
    ]
    return tools

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
    assistant_response: str

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
    """Initialize NvidiaGPT agent with tools"""
    system_prompt = """You are NvidiaGPT, an AI assistant specialized in NVIDIA financial analysis.
    You have access to multiple data sources and tools. Use them wisely to provide comprehensive analysis."""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = create_tools()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        system_message=system_prompt
    )
    
    return agent

def generate_workflow_diagram(filename="nvidia_workflow"):
    """Generates and saves workflow diagram with visual enhancements"""
    dot = Digraph(comment='NVIDIA Analysis Pipeline')
    
    # Set global graph attributes
    dot.attr(rankdir='LR', bgcolor='white', fontname='Helvetica')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', fontcolor='white', margin='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10', penwidth='1.5')
    
    # Add nodes with colors and shapes
    dot.node('start', 'Start', shape='oval', style='filled', fillcolor='#4CAF50', color='#2E7D32')
    dot.node('web_search', 'Web Search', shape='box', style='filled,rounded', fillcolor='#2196F3', color='#0D47A1')
    dot.node('rag_search', 'RAG Search', shape='box', style='filled,rounded', fillcolor='#03A9F4', color='#0277BD')
    dot.node('snowflake', 'Snowflake', shape='box', style='filled,rounded', fillcolor='#00BCD4', color='#006064')
    dot.node('agent', 'NvidiaGPT Agent', shape='hexagon', style='filled', fillcolor='#9C27B0', color='#4A148C')
    dot.node('report_generator', 'Report Generator', shape='note', style='filled', fillcolor='#FF9800', color='#E65100')
    dot.node('end', 'End', shape='oval', style='filled', fillcolor='#F44336', color='#B71C1C')
    
    # Add edges with colors
    dot.edge('start', 'web_search', color='#2196F3')
    dot.edge('start', 'rag_search', color='#03A9F4')
    dot.edge('start', 'snowflake', color='#00BCD4')
    dot.edge('web_search', 'agent', color='#2196F3')
    dot.edge('rag_search', 'agent', color='#03A9F4')
    dot.edge('snowflake', 'agent', color='#00BCD4')
    dot.edge('agent', 'report_generator', color='#9C27B0')
    dot.edge('report_generator', 'end', color='#FF9800')
    
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
    
    # Initialize NvidiaGPT agent
    nvidia_gpt = initialize_nvidia_gpt()
    
    # Define agent node function
    def agent_node(state: NvidiaGPTState) -> Dict:
        """Execute NvidiaGPT agent with current state"""
        # Prepare context from previous nodes
        context = (
            f"Web Search Results: {state.get('web_output', 'No data')}\n"
            f"RAG Search Results: {state.get('rag_output', 'No data')}\n"
            f"Snowflake Query Results: {state.get('snowflake_output', 'No data')}\n"
        )
        
        # Run agent with context
        response = nvidia_gpt.run(
            f"{context}\n\nBased on this information, {state['question']}"
        )
        
        return {"assistant_response": response}
    
    # Add nodes with proper callable functions
    graph.add_node("start", start_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("snowflake", snowflake_node)
    graph.add_node("agent", agent_node)  # Add NvidiaGPT agent node
    graph.add_node("report_generator", final_report_node)
    
    # Set flow
    graph.set_entry_point("start")
    graph.add_edge("start", "web_search")
    graph.add_edge("start", "rag_search")
    graph.add_edge("start", "snowflake")
    graph.add_edge("web_search", "agent")  # Route through agent
    graph.add_edge("rag_search", "agent")
    graph.add_edge("snowflake", "agent")
    graph.add_edge("agent", "report_generator")
    graph.add_edge("report_generator", END)
    
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
        
        print("\nAnalysis Results:\nAccessing Snowflake, RAG and Web Agents")
        print(result.get("final_report"))
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())