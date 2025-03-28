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
st.sidebar.markdown("### Search Configuration")
search_type = st.sidebar.radio(
    "Select Search Type",
    options=["All Quarters", "Specific Quarter"],
    key="search_type"
)

if search_type == "Specific Quarter":
    # Generate all year-quarter combinations from 2020q1 to 2025q4
    quarter_options = [f"{year}q{quarter}" for year in range(2020, 2026) for quarter in range(1, 5)]
    
    selected_periods = st.sidebar.multiselect(
        "Select Period(s)",
        options=quarter_options,
        default=["2023q1"],
        key="period_select"
    )
    
    # Set default if nothing selected
    if not selected_periods:
        selected_periods = ["2023q1"]
    
    # Extract year and quarter from the first selection for compatibility with existing code
    first_selection = selected_periods[0]
    st.session_state.year_slider = first_selection.split('q')[0]
    st.session_state.quarter_slider = first_selection.split('q')[1]
else:
    st.session_state.year_slider = "all"
    st.session_state.quarter_slider = "all"

# Initialize session state for navigation if not already set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Inject custom CSS
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        font-weight: bold;
        height: 3em;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    
    .visualization-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .chat-container {
        margin-bottom: 30px;
        height: 60vh;
        overflow-y: auto;
    }
    
    .user-message {
        background-color: #2196F3;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
    }
    
    .assistant-message {
        background-color: #262730;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
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

    /* Make the submit button a small circle, ChatGPT-style */
    [data-testid="stFormSubmitButton"] button {
        border-radius: 50%;
        width: 50px;
        height: 50px;
        padding: 0;
        min-width: 0;
        font-size: 1.4em;  /* for a small icon or text */
        font-weight: bold;
        background-color: #2196F3;
        color: #fff;
        border: none;
        transition: background-color 0.3s ease;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #0b79d0;
    }
    /* Radio button styling */
    .stRadio > label {
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .stRadio > div {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Slider styling when radio is specific */
    .stSelectSlider {
        margin-top: 15px;
    }
     /* Report styling */
    .report-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    
    .report-section {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .metric-label {
        font-size: 14px;
        color: #B0B0B0;
        margin-top: 5px;
    }
    
    /* Chart styling */
    .chart-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Create navigation buttons in sidebar
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
    
    - **RAG Agent:** Retrieves historical quarterly reports from Pinecone (Year/Quarter).
    - **Web Search Agent:** Provides real-time insights via SerpAPI.
    - **Snowflake Agent:** Queries structured valuation metrics from Snowflake and displays charts.
    
    Use the navigation panel to generate a combined research report or learn more about the application.
    """)

# -------------------------------
# Combined Research Report Page
# -------------------------------
elif page == "Combined Report":
    st.title("NVIDIA Research Assistant")
    
    # Chat container
    chat_container = st.container()
    
    # Display visualization for snowflake data - separate container
    viz_container = st.container()
    
    with chat_container:
        st.markdown("### Research History")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                # Show user message with selected periods
                periods_text = "All Quarters" if message.get('search_type') == "All Quarters" else ", ".join(message.get('selected_periods', []))
                st.markdown(f"""
                <div class="user-message">
                    <div class="metadata">📅 {periods_text}</div>
                    <div>🔍 {message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show assistant message with report structure
                final_report = message.get("content", "")
                
                # Show RAG results if available
                rag_output = message.get("rag_output", {})
                if rag_output:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="metadata">🤖 NVIDIA Research Assistant</div>
                        <div>{final_report}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show Snowflake visualization if available
                if message.get("snowflake_data") and viz_container:
                    with viz_container:
                        st.markdown("### Financial Metrics")
                        if "chart" in message["snowflake_data"]:
                            st.image(
                                f"data:image/png;base64,{message['snowflake_data']['chart']}", 
                                caption="NVIDIA Financial Metrics"
                            )
                        st.markdown("#### Key Metrics")
                        if "metrics" in message["snowflake_data"]:
                            metrics = message["snowflake_data"]["metrics"]
                            st.dataframe(metrics)
    
    # Input form at the bottom
    st.markdown("---")
    with st.form(key="report_form", clear_on_submit=True):
        question = st.text_input(
            "Research Question",
            placeholder="What has driven NVIDIA's revenue growth in recent quarters?",
            key="question_input"
        )
        
        # Use session state values from sidebar
        search_type = st.session_state.search_type
        selected_periods = st.session_state.get("period_select", ["2023q1"]) if search_type == "Specific Quarter" else ["all"]
        
        submitted = st.form_submit_button("🔍", use_container_width=True)
    
    if submitted and question:
        with st.spinner("🤖 Generating comprehensive NVIDIA analysis..."):
            payload = {
                "question": question,
                "search_type": search_type,
                "selected_periods": selected_periods
            }
            
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add to chat history with full structured data
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question,
                        "search_type": search_type,
                        "selected_periods": selected_periods
                    })
                    
                    # Add assistant response with all data
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data.get("final_report", "No report generated"),
                        "rag_output": data.get("rag_output", {}),
                        "snowflake_data": data.get("valuation_data", {})
                    })
                    
                    # Refresh the UI
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
    # Display overall metrics and KPIs at the bottom
    if st.session_state.chat_history and any("snowflake_data" in msg for msg in st.session_state.chat_history if msg["role"] == "assistant"):
        st.markdown("---")
        st.markdown("### NVIDIA Key Performance Indicators")
        latest_data = next((msg["snowflake_data"] for msg in reversed(st.session_state.chat_history) 
                          if msg["role"] == "assistant" and "snowflake_data" in msg), {})
        
        if latest_data and "metrics" in latest_data:
            # Create metrics display
            metrics = latest_data["metrics"]
            if metrics and isinstance(metrics, list) and len(metrics) > 0:
                cols = st.columns(4)
                for i, (key, value) in enumerate(metrics[0].items()[:8]):  # Show first 8 metrics
                    if key != "DATE":
                        with cols[i % 4]:
                            st.metric(label=key, value=value)

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("About NVIDIA Research Assistant")
    st.markdown("""
    **NVIDIA Multi-Agent Research Assistant** integrates:
    
    - **RAG Agent:** Uses Pinecone (index: `nvidia-reports`) with metadata filtering 
      (e.g., `2023q2`, `2024q1`) for historical quarterly reports.
    - **Web Search Agent:** Uses SerpAPI for real-time web search related to NVIDIA.
    - **Snowflake Agent:** Connects to Snowflake to query structured NVIDIA valuation measures and displays visual charts.
    
    **Usage Instructions:**
    - Go to the **Combined Report** page to generate a comprehensive research report.
    - Adjust the Year and Quarter sliders in the sidebar.
    - Enter your question at the bottom, then click the circular arrow button to submit.
    
    **Developed by:** Your Team Name
    """)

