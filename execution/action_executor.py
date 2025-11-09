# execution/action_executor.py (UPDATED)
"""
Action Executor - Now with Tool Support
"""

from typing import Dict, Optional, List
import time
from core.agent_context import AgentContext, ExecutionStatus
from execution.base import ExecutionSystem
from execution.tools.tool_registry import ToolRegistry

class ActionExecutor(ExecutionSystem):
    """Execute actions using tools from registry"""
    
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tool_registry = tool_registry or ToolRegistry()
        self.execution_history = []
    
    def execute(self, context: AgentContext) -> AgentContext:
        """Execute action from reasoning phase"""
        context.log_step("Execution", "Starting execution phase")
        
        # Get action from reasoning
        action = context.get_next_action()
        
        if not action:
            context.log_step("Execution", "No actions to execute")
            context.execution.action_status = ExecutionStatus.PENDING
            return context
        
        context.log_step("Execution", f"Executing: {action}")
        
        try:
            # Extract tool name and params
            tool_name = action.get("tool", "")
            params = action.get("params", {})
            
            if not tool_name:
                raise ValueError("Action missing 'tool' field")
            
            # Execute via tool registry
            result = self.tool_registry.execute_tool(tool_name, **params)
            
            # Update context
            context.execution.action = action
            context.execution.result = result.output
            context.execution.action_status = (
                ExecutionStatus.COMPLETED if result.success 
                else ExecutionStatus.FAILED
            )
            context.execution.error = result.error
            context.execution.execution_time_ms = result.execution_time_ms
            
            context.log_step(
                "Execution",
                f"Action completed: {'✓' if result.success else '✗'}"
            )
        
        except Exception as e:
            context.execution.action_status = ExecutionStatus.FAILED
            context.execution.error = str(e)
            context.log_step("Execution", f"Error: {str(e)}", level="ERROR")
        
        return context
    
    def validate_action(self, action: Dict) -> bool:
        """Validate action has correct structure"""
        if not isinstance(action, dict):
            return False
        
        tool_name = action.get("tool")
        if not tool_name:
            return False
        
        if not self.tool_registry.has_tool(tool_name):
            return False
        
        return True
    
    def register_tool(self, tool):
        """Register a new tool"""
        self.tool_registry.register(tool)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return self.tool_registry.list_tools()
    
    def execute_tool(self, tool_name: str, **params):
        """Execute specific tool (abstract method implementation)"""
        from execution.base import ActionResult
        
        # Execute via registry
        tool_result = self.tool_registry.execute_tool(tool_name, **params)
        
        # Convert ToolResult to ActionResult
        return ActionResult(
            action_id=f"{tool_name}_{int(time.time()*1000)}",
            status="success" if tool_result.success else "failure",
            output=tool_result.output,
            error=tool_result.error,
            execution_time_ms=tool_result.execution_time_ms
        )
    
    def handle_error(self, error: Exception, action: Dict):
        """Handle execution errors (abstract method implementation)"""
        from execution.base import ActionResult
        return ActionResult(
            action_id=str(time.time()),
            status="failure",
            output=None,
            error=str(error),
            execution_time_ms=0.0
        )
