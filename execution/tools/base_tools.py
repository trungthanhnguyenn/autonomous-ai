# execution/tools/base_tool.py
"""
Abstract Tool Base Class
All tools inherit from this
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    """Status of tool execution"""
    AVAILABLE = "available"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str              # "string", "int", "float", "bool", "list", "dict"
    description: str
    required: bool = True
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None  # For enum-like parameters
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate parameter value"""
        # Type checking
        type_map = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Expected {self.type}, got {type(value).__name__}"
        
        # Choice checking
        if self.choices and value not in self.choices:
            return False, f"Value must be one of {self.choices}"
        
        return True, None


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    output: Any                          # Result data
    error: Optional[str] = None          # Error message
    execution_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


class BaseTool(ABC):
    """
    Abstract Base Class for all Tools
    
    Tools are actions the agent can perform:
    - Search information
    - Navigate to URLs
    - Execute code
    - Call APIs
    - etc.
    """
    
    def __init__(self, name: str, description: str):
        """
        Args:
            name: Unique tool name (e.g., "search", "navigate")
            description: What this tool does
        """
        self.name = name
        self.description = description
        self.parameters: List[ToolParameter] = []
        self.status = ToolStatus.AVAILABLE
    
    # ========================================================================
    # ABSTRACT METHODS - Must implement in subclass
    # ========================================================================
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        Return list of parameters this tool accepts
        
        Example:
        return [
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("max_results", "int", "Max results", required=False, default=5)
        ]
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool
        
        Args:
            **kwargs: Parameters as defined in get_parameters()
            
        Returns:
            ToolResult with output or error
        """
        pass
    
    # ========================================================================
    # VALIDATION & DESCRIPTION
    # ========================================================================
    
    def validate_parameters(self, params: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate all parameters before execution
        
        Args:
            params: Parameter dictionary
            
        Returns:
            (is_valid, error_message)
        """
        self.parameters = self.get_parameters()
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
        
        # Validate each parameter
        for param_name, param_value in params.items():
            # Find parameter definition
            param_def = next(
                (p for p in self.parameters if p.name == param_name),
                None
            )
            
            if param_def is None:
                return False, f"Unknown parameter: {param_name}"
            
            # Validate value
            is_valid, error = param_def.validate(param_value)
            if not is_valid:
                return False, f"Parameter '{param_name}': {error}"
        
        return True, None
    
    def get_schema(self) -> Dict:
        """
        Get tool schema for LLM to understand
        
        Returns:
            Tool definition that can be passed to LLM
        """
        self.parameters = self.get_parameters()
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "choices": p.choices,
                }
                for p in self.parameters
            ]
        }
    
    def get_description_text(self) -> str:
        """Get human-readable tool description"""
        self.parameters = self.get_parameters()
        
        desc = f"Tool: {self.name}\n"
        desc += f"Description: {self.description}\n"
        desc += "Parameters:\n"
        
        for param in self.parameters:
            required_str = "required" if param.required else "optional"
            desc += f"  - {param.name} ({param.type}, {required_str}): {param.description}\n"
        
        return desc
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
    
    def __str__(self) -> str:
        return self.get_description_text()
