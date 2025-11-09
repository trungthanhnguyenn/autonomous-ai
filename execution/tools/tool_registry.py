# execution/tools/tool_registry.py
"""
Tool Registry & Manager
Centralized tool management for all systems
"""

from typing import Dict, Optional, List, Any
from execution.tools.base_tools import BaseTool, ToolResult
import json


class ToolRegistry:
    """
    Central registry for all available tools
    
    Manages:
    - Tool registration
    - Tool lookup
    - Tool execution
    - Tool schema for LLM
    """
    
    def __init__(self):
        """Initialize empty registry"""
        self.tools: Dict[str, BaseTool] = {}
        self.execution_history: List[Dict] = []
    
    # ========================================================================
    # REGISTRATION
    # ========================================================================
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool
        
        Args:
            tool: Tool instance
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def register_many(self, tools: List[BaseTool]) -> None:
        """Register multiple tools"""
        for tool in tools:
            self.register(tool)
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            print(f"✓ Unregistered tool: {tool_name}")
        else:
            raise ValueError(f"Tool '{tool_name}' not found")
    
    # ========================================================================
    # LOOKUP
    # ========================================================================
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if tool exists"""
        return tool_name in self.tools
    
    def list_tools(self) -> List[str]:
        """Get list of registered tool names"""
        return list(self.tools.keys())
    
    def get_tool_count(self) -> int:
        """Get number of registered tools"""
        return len(self.tools)
    
    # ========================================================================
    # SCHEMA & DESCRIPTION
    # ========================================================================
    
    def get_all_schemas(self) -> List[Dict]:
        """
        Get schemas of all tools for LLM
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get schema of specific tool"""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_schema()
        return None
    
    def get_all_descriptions(self) -> str:
        """Get descriptions of all tools for LLM context"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_description_text())
        
        return "\n".join(descriptions)
    
    def get_tool_descriptions_json(self) -> str:
        """Get tool descriptions as JSON"""
        return json.dumps(self.get_all_schemas(), indent=2)
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    def execute_tool(self, 
                    tool_name: str,
                    **params) -> ToolResult:
        """
        Execute a tool
        
        Args:
            tool_name: Name of tool to execute
            **params: Tool parameters
            
        Returns:
            ToolResult
        """
        # Get tool
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Validate parameters
        is_valid, error = tool.validate_parameters(params)
        if not is_valid:
            return ToolResult(
                success=False,
                output=None,
                error=error
            )
        
        # Execute tool
        try:
            result = tool.execute(**params)
        except Exception as e:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}"
            )
        
        # Record in history
        self.execution_history.append({
            "tool": tool_name,
            "params": params,
            "result": result.to_dict()
        })
        
        return result
    
    # ========================================================================
    # HISTORY & ANALYSIS
    # ========================================================================
    
    def get_execution_history(self, tool_name: Optional[str] = None) -> List[Dict]:
        """Get execution history"""
        if tool_name:
            return [h for h in self.execution_history if h["tool"] == tool_name]
        return self.execution_history
    
    def get_tool_usage_stats(self) -> Dict:
        """Get statistics about tool usage"""
        stats = {}
        for tool_name in self.list_tools():
            executions = self.get_execution_history(tool_name)
            successes = sum(1 for e in executions if e["result"]["success"])
            
            stats[tool_name] = {
                "total_executions": len(executions),
                "successful": successes,
                "failed": len(executions) - successes,
                "success_rate": successes / len(executions) if executions else 0
            }
        
        return stats
    
    # ========================================================================
    # DEBUG
    # ========================================================================
    
    def print_registry(self):
        """Print all registered tools"""
        print("\n" + "="*70)
        print(f"🔧 TOOL REGISTRY - {self.get_tool_count()} tools registered")
        print("="*70)
        print(self.get_all_descriptions())
        print("="*70)
    
    def print_usage_stats(self):
        """Print tool usage statistics"""
        stats = self.get_tool_usage_stats()
        
        print("\n" + "="*70)
        print("📊 TOOL USAGE STATISTICS")
        print("="*70)
        
        for tool_name, stat in stats.items():
            print(f"\n{tool_name}:")
            print(f"  Total executions: {stat['total_executions']}")
            print(f"  Successful: {stat['successful']}")
            print(f"  Failed: {stat['failed']}")
            print(f"  Success rate: {stat['success_rate']:.1%}")
        
        print("="*70)
