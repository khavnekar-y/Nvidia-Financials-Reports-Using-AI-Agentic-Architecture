from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import requests
import time
import re
import hashlib
import pathlib

load_dotenv()

# Jina AI Reader service prefix
JINA_PREFIX = "https://r.jina.ai/"

class NvidiaWebSearchAgent:
    """
    Agent for retrieving real-time web information about Nvidia using SerpAPI.
    This agent can search for specific topics, latest news, and financial information.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError("SERP_API_KEY not found in environment variables")
        
        # Create a directory for storing markdown content if it doesn't exist
        self.content_dir = pathlib.Path("WebAgent/content")
        self.content_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store links from different searches
        self.nvidia_links = {
            "general": [],
            "news": [],
            "financial": [],
            "quarterly": []
        }
    
    def search(self, query: str, num_results: int = 5, location: str = "United States") -> Dict[str, Any]:
        """
        Perform a general web search with the given query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            location: Location for search context
            
        Returns:
            Dictionary containing processed search results
        """
        params = {
            "q": query,
            "location": location,
            "api_key": self.api_key,
            "hl": "en",
            "num": num_results,
            "gl": "us"  # Country to use for the search
        }
        
        search = GoogleSearch(params)
        raw_results = search.get_dict()
        
        # Process and extract relevant information
        processed_results = {
            "query": query,
            "search_timestamp": datetime.now().isoformat(),
            "organic_results": self._extract_organic_results(raw_results, num_results),
        }
        
        if "news_results" in raw_results:
            processed_results["news_results"] = self._extract_news_results(raw_results, num_results)
        
        # Store links in nvidia_links dictionary
        if "organic_results" in processed_results:
            self.nvidia_links["general"] = [result["link"] for result in processed_results["organic_results"]]
            
        return processed_results
    
    def search_news(self, query: str = "nvidia", num_results: int = 5) -> Dict[str, Any]:
        """
        Search specifically for recent news related to Nvidia.
        
        Args:
            query: Search query string, defaults to "nvidia"
            num_results: Number of results to return
            
        Returns:
            Dictionary containing news results
        """
        params = {
            "q": f"{query} news",
            "tbm": "nws",  # News tab
            "api_key": self.api_key,
            "hl": "en",
            "num": num_results
        }
        
        search = GoogleSearch(params)
        raw_results = search.get_dict()
        
        results = {
            "query": f"{query} news",
            "search_timestamp": datetime.now().isoformat(),
            "news_results": self._extract_news_results(raw_results, num_results)
        }
        
        # Store links in nvidia_links dictionary
        if "news_results" in results:
            self.nvidia_links["news"] = [result["link"] for result in results["news_results"]]
            
        return results
    
    def search_financial_info(self, specific_topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for Nvidia financial information, optionally focused on a specific topic.
        
        Args:
            specific_topic: Optional specific financial topic (earnings, revenue, etc.)
            
        Returns:
            Dictionary containing financial information results
        """
        query = "nvidia financial" if not specific_topic else f"nvidia {specific_topic} financial"
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us"
        }
        
        search = GoogleSearch(params)
        raw_results = search.get_dict()
        
        results = {
            "query": query,
            "search_timestamp": datetime.now().isoformat(),
            "financial_results": self._extract_organic_results(raw_results, 5)
        }
        
        # Store links in nvidia_links dictionary
        if "financial_results" in results:
            self.nvidia_links["financial"] = [result["link"] for result in results["financial_results"]]
            
        return results
    
    def search_quarterly_report_info(self, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for information about Nvidia quarterly reports, optionally filtered by year and quarter.
        
        Args:
            year: Optional year to filter results
            quarter: Optional quarter to filter results (1-4)
            
        Returns:
            Dictionary containing quarterly report information
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
            "gl": "us"
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
        
        # Store links in nvidia_links dictionary
        if "results" in results:
            self.nvidia_links["quarterly"] = [result["link"] for result in results["results"]]
            
        return results
        
    def _extract_organic_results(self, raw_results: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Extract and clean up organic search results.
        """
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
    
    def _extract_news_results(self, raw_results: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Extract and clean up news results.
        """
        results = []
        news_results = raw_results.get("news_results", [])
        if not news_results and "organic_results" in raw_results:
            # Fall back to organic results if no specific news results
            return self._extract_organic_results(raw_results, limit)
            
        for item in news_results[:limit]:
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            if "source" in item:
                result["source"] = item["source"]
            if "date" in item:
                result["date"] = item["date"]
            results.append(result)
        return results
    
    def fetch_url_content(self, url: str) -> str:
        """
        Fetch content from a URL using jina.ai reader service.
        
        Args:
            url: The original URL to fetch
            
        Returns:
            str: The content of the URL in markdown format
        """
        # Prepend the jina.ai prefix to the URL
        jina_url = JINA_PREFIX + url
        
        try:
            # Make the request to jina.ai reader service
            response = requests.get(jina_url)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.text
            else:
                print(f"Error fetching URL: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching URL: {e}")
            return None
    
    def generate_filename(self, url: str, prefix: str) -> str:
        """Generate a filename from URL."""
        # Create a short hash of the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Remove special characters and limit length
        clean_url = re.sub(r'[^\w\s-]', '', url)
        clean_url = re.sub(r'[\s-]+', '_', clean_url)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{prefix}_{timestamp}_{url_hash}.md"
        
        return filename
    
    def save_content_as_markdown(self, content: str, url: str, category: str) -> str:
        """Save content as markdown file with metadata."""
        if not content:
            return None
            
        # Generate filename
        filename = self.generate_filename(url, category)
        filepath = self.content_dir / filename
        
        # Add metadata at the top of the markdown file
        metadata = f"""---
source_url: {url}
category: {category}
fetch_date: {datetime.now().isoformat()}
---

# Content from {url}

"""
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(metadata + content)
            
        return str(filepath)
    
    def fetch_and_save_all_content(self) -> Dict[str, List[str]]:
        """
        Fetch and save content from all stored links.
        
        Returns:
            Dict with categories and lists of saved file paths
        """
        saved_files = {
            "general": [],
            "news": [],
            "financial": [],
            "quarterly": []
        }
        
        # Process all categories
        for category, links in self.nvidia_links.items():
            print(f"Fetching {len(links)} {category} links...")
            
            for link in links:
                try:
                    print(f"Fetching: {link}")
                    content = self.fetch_url_content(link)
                    
                    if content:
                        filepath = self.save_content_as_markdown(content, link, category)
                        if filepath:
                            saved_files[category].append(filepath)
                            print(f"Saved: {filepath}")
                    
                    # Be nice to the Jina service with a small delay
                    time.sleep(1)
                except Exception as e:
                    print(f"Error processing {link}: {e}")
        
        return saved_files


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = NvidiaWebSearchAgent()
    
    # Example 1: General search for Nvidia reports
    general_results = agent.search("nvidia quarterly financial reports", num_results=3)
    print("GENERAL SEARCH RESULTS:")
    print(json.dumps(general_results, indent=2))
    print("\n" + "-"*80 + "\n")
    
    # Example 2: Get latest news about Nvidia
    news_results = agent.search_news(num_results=3)
    print("NEWS SEARCH RESULTS:")
    print(json.dumps(news_results, indent=2))
    print("\n" + "-"*80 + "\n")
    
    # Example 3: Get information about a specific quarterly report
    quarterly_results = agent.search_quarterly_report_info(year=2023, quarter=4)
    print("QUARTERLY REPORT SEARCH RESULTS:")
    print(json.dumps(quarterly_results, indent=2))
    print("\n" + "-"*80 + "\n")
    
    # Fetch and save content from all search results
    print("\nFetching and saving content from all search results...")
    saved_files = agent.fetch_and_save_all_content()
    
    # Print summary of saved files
    print("\nSUMMARY OF SAVED FILES:")
    for category, files in saved_files.items():
        if files:
            print(f"\n{category.upper()} FILES:")
            for file in files:
                print(f"  - {file}")