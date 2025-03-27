import streamlit as st
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import toml

# -------------------------------
# Configuration
# -------------------------------

config = toml.load("config.toml")
API_URL = config["fastapi_url"]
QUERY_URL = f"{API_URL}/research_report"

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("NVIDIA Research Assistant")
# Initialize session state for navigation if not already set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS to make buttons more visually pleasing
st.sidebar.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        font-weight: bold;
        height: 3em;
        margin-bottom: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Create navigation buttons
for page_name in ["Home", "Combined Report", "About"]:
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        type="primary" if st.session_state.current_page == page_name else "secondary",
        use_container_width=True
    ):
        st.session_state.current_page = page_name
        st.rerun()

# Set the current page from session state
page = st.session_state.current_page

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("Welcome to the NVIDIA Multi-Agent Research Assistant")
    st.markdown("""
    This application integrates multiple agents to produce comprehensive research reports on NVIDIA:
    
    - **RAG Agent:** Retrieves historical quarterly reports from Pinecone with metadata filtering (Year/Quarter).
    - **Web Search Agent:** Provides real-time insights via SerpAPI.
    - **Snowflake Agent:** Queries structured valuation metrics from Snowflake and displays charts.
    
    Use the navigation panel to generate a combined research report or learn more about the application.
    """)

# -------------------------------
# Combined Research Report Page
# -------------------------------
elif page == "Combined Report":
    st.title("Combined Research Report")
    # Custom CSS for chat interface and report styling
    st.markdown("""
    <style>
        .chat-container {
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #2196F3;
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            color: white;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .assistant-message {
            background-color: #262730;
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            color: white;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .metadata {
            font-size: 0.8em;
            color: #B0B0B0;
            margin-bottom: 5px;
        }
        .report-section {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .report-header {
            color: #4CAF50;
            font-size: 1.2em;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history
    st.markdown("### Previous Conversations")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="metadata">📅 {message['year']}Q{message['quarter']}</div>
                <div>🔍 {message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="metadata">🤖 NVIDIA Research Assistant</div>
                <div>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input form with improved styling
    st.markdown("### New Research Query")
    with st.form(key="report_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            question = st.text_input(
                "Research Question",
                placeholder="E.g., What are the key factors affecting NVIDIA's performance?",
                key="question_input"
            )
        with col2:
            year = st.number_input("Year", min_value=2020, max_value=2025, value=2023)
        with col3:
            quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0)
        
        col_submit, col_clear = st.columns([4, 1])
        with col_submit:
            submitted = st.form_submit_button(
                "🔍 Generate Research Report",
                use_container_width=True,
                type="primary"
            )
        with col_clear:
            clear_history = st.form_submit_button(
                "🗑️ Clear History",
                use_container_width=True,
                type="secondary"
            )
    
    if clear_history:
        st.session_state.chat_history = []
        st.rerun()
    
    if submitted and question:
        with st.spinner("🤖 Analyzing NVIDIA data..."):
            payload = {"question": question, "year": year, "quarter": quarter}
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    final_report = data.get("final_report", "No report generated.")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "year": year,
                        "quarter": quarter
                    })
                    
                    # Format and display the report
                    formatted_report = f"""
                    <div class="report-section">
                        <div class="report-header">📊 Research Report</div>
                        <div>{final_report}</div>
                    </div>
                    """
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": formatted_report
                    })
                    
                    st.rerun()
                else:
                    st.error("🚫 Backend error: " + response.text)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Add download button for chat history
    if st.session_state.chat_history:
        chat_history_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            label="📥 Download Conversation History",
            data=chat_history_json,
            file_name="nvidia_research_history.json",
            mime="application/json"
        )

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("About NVIDIA Research Assistant")
    st.markdown("""
    **NVIDIA Multi-Agent Research Assistant** integrates:
    
    - **RAG Agent:** Uses Pinecone (index: `nvidia-reports`) with metadata filtering (namespaces like `2023q2`, `2024q1`, etc.) to retrieve historical NVIDIA quarterly reports.
    - **Web Search Agent:** Uses SerpAPI for real-time web search related to NVIDIA.
    - **Snowflake Agent:** Connects to Snowflake to query structured NVIDIA valuation measures and displays visual charts.
    
    The backend is implemented with FastAPI (see the provided `main.py`), and this frontend is built with Streamlit for a modern, conversational UI.
    
    **Usage Instructions:**
    - Use the **Combined Report** page to generate a comprehensive research report.
    - Adjust the research question, Year, and Quarter as needed.
    - The app returns a consolidated report that includes historical, real-time, and structured financial insights.
    
    **Developed by:** Your Team Name
    """)

