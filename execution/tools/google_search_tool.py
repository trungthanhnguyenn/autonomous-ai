# execution/tools/google_search_tool.py
"""
Google Search Tool - Real implementation using Google Custom Search API
"""

import os
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from execution.tools.base_tools import BaseTool, ToolParameter, ToolResult

# Load environment variables
load_dotenv()


class GoogleSearchTool(BaseTool):
    """
    Real Google Search using Google Custom Search API
    
    Requires environment variables:
    - GOOGLE_SEARCH_ENGINE_KEY: Google API key
    - GOOGLE_ID_CSE: Custom Search Engine ID
    """
    
    def __init__(self):
        super().__init__(
            name="google_search",
            description="Search the web using Google Custom Search API. Returns real search results with titles, URLs, and snippets."
        )
        
        # Load credentials from environment
        self.api_key = os.getenv("GOOGLE_SEARCH_ENGINE_KEY")
        self.cse_id = os.getenv("GOOGLE_ID_CSE")
        
        if not self.api_key or not self.cse_id:
            print("⚠️  Warning: Google Search credentials not found in .env")
            print("   Set GOOGLE_SEARCH_ENGINE_KEY and GOOGLE_ID_CSE")
        else:
            print(f"✓ Google Search Tool initialized (CSE ID: {self.cse_id[:8]}...)")
    
    def get_parameters(self) -> List[ToolParameter]:
        """Define tool parameters"""
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query to look up on Google",
                required=True
            ),
            ToolParameter(
                name="num_results",
                type="int",
                description="Number of results to return (1-10)",
                required=False,
                default=5
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute Google Search
        
        Args:
            query: Search query string
            num_results: Number of results (default: 5, max: 10)
            
        Returns:
            ToolResult with search results or error
        """
        query = kwargs.get("query", "")
        num_results = min(kwargs.get("num_results", 5), 10)
        
        if not query:
            return ToolResult(
                success=False,
                output=None,
                error="Search query cannot be empty"
            )
        
        # Check credentials
        if not self.api_key or not self.cse_id:
            return ToolResult(
                success=False,
                output=None,
                error="Google Search API credentials not configured"
            )
        
        start_time = time.time()
        
        try:
            # Call Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract search results
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "displayLink": item.get("displayLink", "")
                    })
            
            execution_time = (time.time() - start_time) * 1000
            
            # Format output
            output = {
                "query": query,
                "results": results,
                "total_results": len(results),
                "search_info": data.get("searchInformation", {})
            }
            
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
                metadata={
                    "api": "google_custom_search",
                    "total_results": data.get("searchInformation", {}).get("totalResults", "0")
                }
            )
            
        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                output=None,
                error="Google Search API request timed out",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Google Search API error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for human-readable display
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"   {result['url']}")
            formatted.append(f"   {result['snippet']}")
            formatted.append("")
        
        return "\n".join(formatted)
