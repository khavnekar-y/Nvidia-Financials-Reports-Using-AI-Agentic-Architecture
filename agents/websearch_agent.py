# agents/web_search_agent.py
import os
import json
import pathlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

class NvidiaWebSearchAgent:
    """
    Retrieves real-time web information about NVIDIA using SerpAPI.
    """
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError("SERP_API_KEY not found in environment variables")
        self.content_dir = pathlib.Path("web_agent/content")
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.nvidia_links = {
            "general": [],
            "news": [],
            "financial": [],
            "quarterly": []
        }
    
    def search_quarterly_report_info(self, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for NVIDIA quarterly report information using SerpAPI.
        """
        query = "nvidia quarterly report"
        if year:
            query += f" {year}"
        if quarter and 1 <= quarter <= 4:
            query += f" Q{quarter}"
        params = {
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us",
            "num": 5
        }
        search = GoogleSearch(params)
        raw_results = search.get_dict()
        results = {
            "query": query,
            "search_timestamp": datetime.now().isoformat(),
            "year": year,
            "quarter": quarter,
            "results": self._extract_organic_results(raw_results, 5)
        }
        if "results" in results:
            self.nvidia_links["quarterly"] = [result["link"] for result in results["results"]]
        return results

    def _extract_organic_results(self, raw_results: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        results = []
        if "organic_results" in raw_results and raw_results["organic_results"]:
            for item in raw_results["organic_results"][:limit]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                }
                if "displayed_link" in item:
                    result["source"] = item["displayed_link"]
                results.append(result)
        return results

if __name__ == '__main__':
    # Quick test of the Web Search Agent functionality:
    try:
        agent = NvidiaWebSearchAgent()
        # Example: Retrieve quarterly report information for Q4 2023.
        test_year = 2023
        test_quarter = 4
        results = agent.search_quarterly_report_info(year=test_year, quarter=test_quarter)
        print("Web Search Agent Test Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during testing: {e}")
