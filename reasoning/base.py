# reasoning/base.py
"""
Paper: Reasoning System
"A reasoning system that formulates plans, adapts to feedback, 
and evaluates actions through different techniques like 
Chain-of-Thought and Tree-of-Thought"
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from core.agent_context import AgentContext

@dataclass
class ActionPlan:
    """Structured action plan from reasoning"""
    steps: List[Dict]          # [{"action": "...", "params": {...}}]
    reasoning_trace: List[str] # Step-by-step thinking
    confidence: float          # 0.0-1.0
    alternative_plans: List['ActionPlan'] = None  # For ToT


class ReasoningSystem(ABC):
    """
    Abstract Reasoning System
    
    Paper: "Formulates plans, adapts to feedback, 
    and evaluates actions"
    
    Key Approaches:
    1. Chain-of-Thought: Step-by-step reasoning
    2. Tree-of-Thought: Explore multiple paths
    3. Adaptive Reasoning: Learn from feedback
    """
    
    @abstractmethod
    def reason(self, context: AgentContext) -> AgentContext:
        """
        Main reasoning method
        
        Input: AgentContext with perception + memory data
        Output: ActionPlan with reasoning trace
        
        Paper: "Given perceived environment and memory context,
        generate action plan"
        """
        pass
    
    @abstractmethod
    def chain_of_thought(self, prompt: str) -> Dict:
        """
        Paper: Chain-of-Thought (CoT)
        "LLM explains step-by-step thinking"
        
        Returns: {"thinking": str, "action": str}
        """
        pass
    
    @abstractmethod
    def tree_of_thought(self, prompt: str, num_paths: int = 3) -> Dict:
        """
        Paper: Tree-of-Thought (ToT)
        "Explore multiple reasoning paths, select best"
        
        Returns: {"paths": [], "best_path": int}
        """
        pass
    
    @abstractmethod
    def adapt_to_feedback(self, 
                         previous_action: Dict,
                         feedback: str,
                         context: AgentContext) -> ActionPlan:
        """
        Paper: Adaptive Reasoning
        "Learn from feedback and adjust strategy"
        
        Modify reasoning based on what didn't work
        """
        pass


class PromptBuilder(ABC):
    """Build intelligent prompts for LLM"""
    
    @staticmethod
    @abstractmethod
    def build_reasoning_prompt(context: AgentContext) -> str:
        """
        Build prompt that includes:
        - Current perception
        - Similar past experiences
        - Learned patterns
        - Specific reasoning instructions
        """
        pass
    
    @staticmethod
    @abstractmethod
    def build_cot_prompt(perception: Dict, memory: Dict) -> str:
        """Prompt for Chain-of-Thought"""
        pass
    
    @staticmethod
    @abstractmethod
    def build_tot_prompt(perception: Dict, memory: Dict) -> str:
        """Prompt for Tree-of-Thought"""
        pass
