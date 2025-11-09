# execution/base.py
"""
Paper: Execution System
"Translates internal decisions into concrete actions"
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from core.agent_context import AgentContext

@dataclass
class ActionResult:
    """Result of executing an action"""
    action_id: str
    status: str  # "success", "failure", "partial"
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ExecutionSystem(ABC):
    """
    Abstract Execution System
    
    Paper: "Translates internal decisions 
    (action plans) into concrete actions in the environment"
    
    Responsibilities:
    1. Validate actions are safe/possible
    2. Map to actual tools/APIs
    3. Execute and handle errors
    4. Collect feedback for learning
    """
    
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentContext:
        """
        Execute planned action
        
        Input: AgentContext with action_plan
        Output: AgentContext with execution results
        """
        pass
    
    @abstractmethod
    def validate_action(self, action: Dict) -> bool:
        """Validate action before execution"""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, **params) -> ActionResult:
        """Execute specific tool"""
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception, action: Dict) -> ActionResult:
        """Handle execution errors gracefully"""
        pass
