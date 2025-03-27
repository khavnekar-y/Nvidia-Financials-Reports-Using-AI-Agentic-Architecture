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
    st.markdown("Enter your research question along with the Year and Quarter to generate a comprehensive report.")
    
    with st.form(key="report_form"):
        question = st.text_input("Research Question", 
                                 "What are the key factors affecting NVIDIA's performance?")
        year = st.number_input("Year", min_value=2020, max_value=2025, value=2023)
        quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0)
        submitted = st.form_submit_button("Generate Report")
    
    if submitted:
        with st.spinner("Generating report..."):
            payload = {"question": question, "year": year, "quarter": quarter}
            try:
                response = requests.post(QUERY_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    final_report = data.get("final_report", "No report generated.")
                    
                    st.subheader("Final Research Report")
                    st.markdown(final_report)
                else:
                    st.error("Backend error: " + response.text)
            except Exception as e:
                st.error(f"Error: {e}")

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

