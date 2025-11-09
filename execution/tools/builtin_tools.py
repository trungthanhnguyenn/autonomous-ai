# execution/tools/builtin_tools.py
"""
Built-in Tools - Common tools everyone needs
"""

from typing import List
from execution.tools.base_tools import BaseTool, ToolParameter, ToolResult
import time


class SearchTool(BaseTool):
    """Search for information"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="Search for information using a query"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="int",
                description="Maximum number of results",
                required=False,
                default=5
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute search (simulate)"""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        start_time = time.time()
        
        # Simulate search
        results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/{i}",
                "snippet": f"This is a search result about {query}"
            }
            for i in range(min(max_results, 5))
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            output={"query": query, "results": results},
            execution_time_ms=execution_time
        )


class NavigateTool(BaseTool):
    """Navigate to a URL"""
    
    def __init__(self):
        super().__init__(
            name="navigate",
            description="Navigate to a URL and get page content"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="URL to navigate to",
                required=True
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute navigation (simulate)"""
        url = kwargs.get("url", "")
        
        start_time = time.time()
        
        # Simulate navigation
        page_content = f"Content from {url}\nThis is a simulated page with content."
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=True,
            output={"url": url, "content": page_content},
            execution_time_ms=execution_time
        )


class CodeExecuteTool(BaseTool):
    """Execute Python code"""
    
    def __init__(self):
        super().__init__(
            name="execute_code",
            description="Execute Python code and return result"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Python code to execute",
                required=True
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute code (simulate - don't actually eval)"""
        code = kwargs.get("code", "")
        
        start_time = time.time()
        
        # For safety, simulate instead of actual eval
        try:
            result = f"Executed: {code[:50]}..."
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResult(
            success=success,
            output=result,
            error=error,
            execution_time_ms=execution_time
        )


class ClickTool(BaseTool):
    """Click on UI element"""
    
    def __init__(self):
        super().__init__(
            name="click",
            description="Click on a UI element by selector"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="Element selector (CSS or XPath)",
                required=True
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute click (simulate)"""
        selector = kwargs.get("selector", "")
        
        return ToolResult(
            success=True,
            output={"selector": selector, "status": "clicked"},
            execution_time_ms=10.0
        )


class TypeTool(BaseTool):
    """Type text into input field"""
    
    def __init__(self):
        super().__init__(
            name="type",
            description="Type text into an input field"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="selector",
                type="string",
                description="Input field selector",
                required=True
            ),
            ToolParameter(
                name="text",
                type="string",
                description="Text to type",
                required=True
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute typing (simulate)"""
        selector = kwargs.get("selector", "")
        text = kwargs.get("text", "")
        
        return ToolResult(
            success=True,
            output={"selector": selector, "text": text, "status": "typed"},
            execution_time_ms=len(text) * 10
        )


# Helper function to create default registry
def create_default_tool_registry():
    """Create registry with built-in tools"""
    from execution.tools.tool_registry import ToolRegistry
    
    registry = ToolRegistry()
    registry.register_many([
        SearchTool(),
        NavigateTool(),
        CodeExecuteTool(),
        ClickTool(),
        TypeTool(),
    ])
    
    return registry
