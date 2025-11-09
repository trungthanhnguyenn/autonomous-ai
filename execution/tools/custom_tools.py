# execution/tools/custom_tools.py
"""
Template for creating custom tools
User can extend this for their specific needs
"""

from typing import List
from execution.tools.base_tools import BaseTool, ToolParameter, ToolResult
import time


class CustomTool(BaseTool):
    """
    Template for custom tools
    
    Usage:
        1. Inherit from BaseTool
        2. Define __init__ with name & description
        3. Implement get_parameters()
        4. Implement execute(**kwargs)
    """
    
    def __init__(self, name: str = "custom", description: str = "Custom tool"):
        super().__init__(name, description)
    
    def get_parameters(self) -> List[ToolParameter]:
        """Define what parameters this tool accepts"""
        return [
            ToolParameter(
                name="param1",
                type="string",
                description="First parameter",
                required=True
            ),
            ToolParameter(
                name="param2",
                type="int",
                description="Second parameter",
                required=False,
                default=10
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        param1 = kwargs.get("param1")
        param2 = kwargs.get("param2", 10)
        
        start_time = time.time()
        
        try:
            # Your logic here
            result = f"Custom execution: {param1}, {param2}"
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                output=result,
                execution_time_ms=execution_time
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


# Example: Database query tool
class DatabaseTool(BaseTool):
    """Query a database"""
    
    def __init__(self):
        super().__init__(
            name="database_query",
            description="Execute SQL queries on database"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="SQL query to execute",
                required=True
            ),
            ToolParameter(
                name="database",
                type="string",
                description="Database name",
                required=True,
                choices=["users", "products", "orders"]
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute database query"""
        query = kwargs.get("query")
        database = kwargs.get("database")
        
        # Simulate query
        rows = [
            {"id": 1, "name": "Result 1"},
            {"id": 2, "name": "Result 2"},
        ]
        
        return ToolResult(
            success=True,
            output={"query": query, "database": database, "rows": rows},
            execution_time_ms=50.0
        )


# Example: Email tool
class EmailTool(BaseTool):
    """Send emails"""
    
    def __init__(self):
        super().__init__(
            name="send_email",
            description="Send an email"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="to",
                type="string",
                description="Recipient email",
                required=True
            ),
            ToolParameter(
                name="subject",
                type="string",
                description="Email subject",
                required=True
            ),
            ToolParameter(
                name="body",
                type="string",
                description="Email body",
                required=True
            ),
        ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Send email (simulate)"""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        
        # Simulate sending
        return ToolResult(
            success=True,
            output={"to": to, "subject": subject, "status": "sent"},
            execution_time_ms=100.0
        )
