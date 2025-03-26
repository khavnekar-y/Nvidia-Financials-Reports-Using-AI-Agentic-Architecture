import os
import glob
from typing import List, Dict, Any
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv('.env')

class RAGPineconeAgent:
    def __init__(self, pinecone_api_key=None, pinecone_environment=None, index_name="nvidia-content"):
        """Initialize the RAG Pinecone Agent with API keys and configuration"""
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = index_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing required API keys. Please set PINECONE_API_KEY, PINECONE_ENVIRONMENT, and OPENAI_API_KEY in environment variables.")
        
        # Initialize embedding model - use smaller, more efficient model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        print("Initialized embedding model: all-MiniLM-L6-v2 with dimension 384")
        
    def initialize_pinecone(self):
        """Initialize Pinecone client and create index if it doesn't exist"""
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
        
        # Check if index exists and create it if not
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine"
            )
            print(f"Created new Pinecone index: {self.index_name}")
        
        return pinecone.Index(self.index_name)
    
    def load_markdown_files(self, content_dir: str) -> List[Document]:
        """Load all markdown files from the specified directory into Document objects"""
        markdown_files = glob.glob(os.path.join(content_dir, "*.md"))
        documents = []
        
        for file_path in markdown_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                filename = os.path.basename(file_path)
                # Create metadata with file info
                metadata = {
                    "source": file_path,
                    "filename": filename,
                    "type": filename.split("_")[0]  # Extract type (general, news, quarterly)
                }
                documents.append(Document(page_content=content, metadata=metadata))
                
        print(f"Loaded {len(documents)} markdown documents")
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into smaller chunks for better retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        print(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def create_vectors_and_store(self, documents: List[Document]) -> LangchainPinecone:
        """Create vector embeddings and store them in Pinecone"""
        # Initialize Pinecone
        self.initialize_pinecone()
        
        # Create and store vectors
        vectorstore = LangchainPinecone.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        
        print(f"Created and stored vectors in Pinecone index: {self.index_name}")
        return vectorstore
    
    def setup_rag_pipeline(self, vectorstore: LangchainPinecone):
        """Set up a RAG pipeline using the vector store"""
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Define prompt template
        template = """
        You are an AI assistant specialized in answering questions about NVIDIA.
        Use the following retrieved information to answer the question.
        If you don't know the answer, just say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Define LLM
        llm = ChatOpenAI(temperature=0.2)
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    def build_langgraph_agent(self, vectorstore):
        """Build a LangGraph agent that leverages the RAG system"""
        # Define state
        class AgentState:
            """State for the agent"""
            question: str
            retrieved_docs: List[Document] = None
            answer: str = None
            follow_up_needed: bool = False
            follow_up_question: str = None
            
        # Define nodes
        def retrieve(state: AgentState) -> AgentState:
            """Retrieve relevant documents"""
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            state.retrieved_docs = retriever.invoke(state.question)
            return state
            
        def generate_answer(state: AgentState) -> AgentState:
            """Generate answer based on retrieved docs"""
            llm = ChatOpenAI(temperature=0.2)
            
            context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
            prompt = f"""
            You are an AI assistant specialized in answering questions about NVIDIA.
            Use the following retrieved information to answer the question.
            
            Context:
            {context}
            
            Question: {state.question}
            
            Answer the question based on the context provided. If the context doesn't contain relevant information, 
            say "I don't have enough information to answer this question."
            """
            
            state.answer = llm.invoke(prompt).content
            
            # Determine if a follow-up is needed
            follow_up_prompt = f"""
            Based on the question: "{state.question}" and your answer: "{state.answer}", 
            determine if a follow-up question would be useful to gather more information.
            If yes, respond with "YES: <follow-up question>". If no, respond with "NO".
            """
            
            follow_up_decision = llm.invoke(follow_up_prompt).content
            
            if follow_up_decision.startswith("YES:"):
                state.follow_up_needed = True
                state.follow_up_question = follow_up_decision[4:].strip()
            
            return state
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate_answer", generate_answer)
        
        # Add edges
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_conditional_edges(
            "generate_answer",
            lambda state: "follow_up" if state.follow_up_needed else END,
            {
                "follow_up": "retrieve"
            }
        )
        
        # Set the entry point
        workflow.set_entry_point("retrieve")
        
        # Compile the graph
        agent = workflow.compile()
        
        return agent
    
    def process_content_directory(self, content_dir: str):
        """Process content directory: load files, chunk them, create vectors, and build agent"""
        # Load markdown documents
        documents = self.load_markdown_files(content_dir)
        
        # Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        # Create vectors and store in Pinecone
        vectorstore = self.create_vectors_and_store(chunked_documents)
        
        # Build and return the LangGraph agent
        agent = self.build_langgraph_agent(vectorstore)
        
        return agent

# Main execution
if __name__ == "__main__":
    # Initialize the agent
    rag_agent = RAGPineconeAgent()
    
    # Process the WebAgent content directory
    content_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "WebAgent", "content")
    
    print(f"Processing content from: {content_dir}")
    
    # Build the agent
    agent = rag_agent.process_content_directory(content_dir)
    
    # Example usage
    print("\nAgent created successfully! You can now query the NVIDIA content.")
    while True:
        user_question = input("\nEnter your question about NVIDIA (or 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
            
        # Run the agent
        result = agent.invoke({"question": user_question})
        print("\nAnswer:", result.answer)
        
        if result.follow_up_needed:
            print("\nFollow-up question:", result.follow_up_question)
            follow_up = input("Would you like to ask this follow-up? (y/n): ")
            if follow_up.lower() == "y":
                result = agent.invoke({"question": result.follow_up_question})
                print("\nAnswer:", result.answer)