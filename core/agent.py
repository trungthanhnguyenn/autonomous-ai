# core/agent.py
"""
Main Autonomous Agent Class

Coordinates all 4 systems through the orchestrator.
Manages agent lifecycle and learning.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from core.agent_context import AgentContext, PerceptionModality, create_initial_context
from orchestration.agent_orchestrator import AgentOrchestrator
from perception.base import PerceptionSystem
from reasoning.base import ReasoningSystem
from memory.base import MemorySystem
from execution.base import ExecutionSystem


class AutonomousAgent:
    """
    Main Agent Class
    
    Integrates 4 core systems:
    - Perception: Convert input to understanding
    - Memory: Provide context from past
    - Reasoning: Generate action plans
    - Execution: Execute actions
    
    Paper-Aligned Flow:
    Input → Perception → Memory(retrieve) → Reasoning → 
    Execution → Memory(store) → Output + Feedback
    """
    
    def __init__(
        self,
        llm: Any,
        perception: PerceptionSystem,
        reasoning: ReasoningSystem,
        memory: MemorySystem,
        execution: ExecutionSystem,
        max_iterations: int = 5,
        agent_name: str = "AutonomousAgent"
    ):
        """
        Initialize agent with all systems
        
        Args:
            llm: LLM client (OpenAI, Anthropic, etc.)
            perception: Perception system instance
            reasoning: Reasoning system instance
            memory: Memory system instance
            execution: Execution system instance
            max_iterations: Max steps before stopping
            agent_name: Name for logging
        """
        # Store systems
        self.llm = llm
        self.perception = perception
        self.reasoning = reasoning
        self.memory = memory
        self.execution = execution
        self.max_iterations = max_iterations
        self.agent_name = agent_name
        
        # Create orchestrator
        self.orchestrator = AgentOrchestrator(
            perception=perception,
            reasoning=reasoning,
            memory=memory,
            execution=execution
        )
        
        # Initialize context (will be set when agent runs)
        self.context: Optional[AgentContext] = None
        
        # Track execution history
        self.execution_history: List[Dict] = []
        self.is_running = False
    
    # ========================================================================
    # EXECUTION METHODS
    # ========================================================================
    
    def step(self, env_input: Any, 
             modality: PerceptionModality = PerceptionModality.TEXT) -> Dict[str, Any]:
        """
        Execute one agent step
        
        Single iteration through all 4 systems
        
        Args:
            env_input: Input from environment
            modality: How to interpret input
            
        Returns:
            Result of the step
        """
        # Create context if first step
        if self.context is None:
            self.context = create_initial_context(env_input, modality)
            print(f"\n{self.agent_name} initialized with {modality.value} input")
        else:
            # Reset context for new step but keep history
            self.context.reset()
            self.context.perception.raw_input = env_input
            self.context.perception.modality = modality
        
        # === Execute one step ===
        self.context = self.orchestrator.execute_step(self.context, env_input)
        
        # Record step result
        step_result = self._record_step()
        
        return step_result
    
    def run(self, env_input: Any,
            modality: PerceptionModality = PerceptionModality.TEXT,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Run agent until completion or max iterations
        
        Args:
            env_input: Initial input from environment
            modality: Input modality
            verbose: Print progress
            
        Returns:
            Final result
        """
        self.is_running = True
        
        try:
            # Initialize context
            self.context = create_initial_context(env_input, modality)
            current_input = env_input
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"{self.agent_name} Starting Agent Loop")
                print(f"{'='*70}\n")
            
            # === Main loop ===
            for iteration in range(1, self.max_iterations + 1):
                if verbose:
                    print(f"[Iteration {iteration}/{self.max_iterations}]")
                
                # Execute step
                self.context = self.orchestrator.execute_step(
                    self.context, 
                    current_input
                )
                
                # Record step
                step_result = self._record_step()
                
                if verbose:
                    print(f"  Status: {self.context.execution.action_status.value}")
                    print(f"  Confidence: {self.context.get_reasoning_confidence():.1%}")
                    print()
                
                # Check if should continue
                if not self._should_continue():
                    self.context.is_complete = True
                    self.context.completion_reason = "Action completed"
                    break
                
                # Prepare input for next iteration (feedback loop)
                current_input = self._prepare_next_input()
            
            # Loop ended
            if not self.context.is_complete:
                self.context.completion_reason = "Max iterations reached"
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Agent Completed: {self.context.completion_reason}")
                print(f"{'='*70}\n")
            
            return self._get_final_result()
        
        finally:
            self.is_running = False
    
    # ========================================================================
    # STATE MANAGEMENT METHODS
    # ========================================================================
    
    def _record_step(self) -> Dict[str, Any]:
        """Record step in history"""
        if self.context is None:
            return {}
        
        step_record = {
            "iteration": self.context.iteration,
            "perception_modality": self.context.perception.modality.value,
            "perception_confidence": self.context.perception.get_confidence(),
            "intent": self.context.get_perceived_intent(),
            "action_plan_size": len(self.context.reasoning.action_plan),
            "reasoning_confidence": self.context.get_reasoning_confidence(),
            "execution_status": self.context.execution.action_status.value,
            "was_successful": self.context.is_action_completed(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(step_record)
        self.context.save_to_history()  # Save to context history
        
        return step_record
    
    def _should_continue(self) -> bool:
        """Decide if agent should continue loop"""
        # Check if context exists
        if self.context is None:
            return False
        
        # Stop if action succeeded
        if self.context.is_action_completed():
            return False
        
        # Stop if max iterations reached (handled in loop)
        # Stop if reasoning not confident
        if self.context.get_reasoning_confidence() < 0.3:
            print("Low confidence, stopping")
            return False
        
        # Otherwise continue
        return True 
    
    def _prepare_next_input(self) -> Dict:
        """
        Prepare input for next iteration (feedback loop)
        
        Next iteration gets feedback from current execution
        """
        if self.context is None:
            return {"type": "feedback", "message": "No context"}
        
        return {
            "type": "feedback",
            "previous_action": self.context.execution.action,
            "previous_result": self.context.execution.result,
            "execution_status": self.context.execution.action_status.value
        }
    
    def _get_final_result(self) -> Dict[str, Any]:
        """Get final result after execution"""
        if self.context is None:
            return {"error": "No context"}
        
        return {
            "total_iterations": self.context.iteration,
            "completion_reason": self.context.completion_reason,
            "was_successful": self.context.is_complete and self.context.execution.was_successful(),
            "final_action": self.context.execution.action,
            "final_result": self.context.execution.result,
            "final_error": self.context.execution.error,
            "reasoning_confidence": self.context.get_reasoning_confidence(),
            "execution_history": self.execution_history
        }
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_state_summary(self) -> Dict:
        """Get current agent state summary"""
        if self.context is None:
            return {"status": "not_initialized"}
        
        return self.context.get_state_summary()
    
    def get_execution_trace(self) -> str:
        """Get formatted execution trace"""
        if self.context is None:
            return "No execution trace"
        
        return self.context.get_execution_trace()
    
    def get_perceived_entities(self) -> List[Dict]:
        """Get entities perceived in current state"""
        if self.context is None:
            return []
        
        return self.context.get_extracted_entities()
    
    def get_action_plan(self) -> List[Dict]:
        """Get current action plan"""
        if self.context is None:
            return []
        
        return self.context.reasoning.action_plan
    
    def get_similar_experiences(self) -> List[Dict]:
        """Get similar past experiences"""
        if self.context is None:
            return []
        
        return self.context.get_similar_experiences()
    
    # ========================================================================
    # DEBUG & VISUALIZATION METHODS
    # ========================================================================
    
    def print_context(self):
        """Pretty print current context"""
        if self.context is None:
            print("No context initialized")
        else:
            print(self.context.pretty_print())
    
    def print_execution_history(self):
        """Print execution history"""
        print("\nEXECUTION HISTORY:")
        print(f"{'Iter':>4} | {'Intent':15} | {'Conf':>5} | {'Status':>10} | {'Success':>7}")
        print("-" * 60)
        
        for step in self.execution_history:
            print(f"{step['iteration']:>4} | "
                  f"{step['intent'][:15]:15} | "
                  f"{step['reasoning_confidence']:>4.0%} | "
                  f"{step['execution_status']:>10} | "
                  f"{'✓' if step['was_successful'] else '✗':>7}")
    
    def print_logs(self):
        """Print execution logs"""
        if self.context is None:
            print("No logs")
        else:
            print("\nEXECUTION LOGS:")
            print(self.get_execution_trace())
    
    def export_context_json(self, include_logs: bool = True) -> str:
        """Export context as JSON"""
        if self.context is None:
            return "{}"
        
        return self.context.to_json(include_logs=include_logs)
    
    # ========================================================================
    # LEARNING & ADAPTATION METHODS
    # ========================================================================
    
    def learn_from_history(self) -> Dict:
        """
        Analyze execution history for learning
        
        Returns:
            Learning insights
        """
        if not self.execution_history:
            return {"insights": "No history to learn from"}
        
        successful = [s for s in self.execution_history if s["was_successful"]]
        failed = [s for s in self.execution_history if not s["was_successful"]]
        
        insights = {
            "total_steps": len(self.execution_history),
            "successful_steps": len(successful),
            "failed_steps": len(failed),
            "success_rate": len(successful) / len(self.execution_history) if self.execution_history else 0,
            "avg_confidence": sum(s["reasoning_confidence"] for s in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "most_common_intent": self._get_most_common_intent(),
        }
        
        return insights
    
    def _get_most_common_intent(self) -> Optional[str]:
        """Get most frequently perceived intent"""
        if not self.execution_history:
            return None
        
        intents = [s["intent"] for s in self.execution_history if s["intent"]]
        if not intents:
            return None
        
        # Count occurrences
        intent_counts: Dict[str, int] = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Return intent with max count
        return max(intent_counts.items(), key=lambda x: x[1])[0]
    
    # ========================================================================
    # RESET METHODS
    # ========================================================================
    
    def reset(self):
        """Reset agent for new session"""
        self.context = None
        self.execution_history = []
        self.is_running = False
        print(f"{self.agent_name} reset")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_agent(llm,
                 perception: PerceptionSystem,
                 reasoning: ReasoningSystem,
                 memory: MemorySystem,
                 execution: ExecutionSystem,
                 **kwargs) -> AutonomousAgent:
    """
    Factory function to create agent
    
    Args:
        llm: LLM client
        perception: Perception system
        reasoning: Reasoning system
        memory: Memory system
        execution: Execution system
        **kwargs: Additional args (max_iterations, agent_name)
        
    Returns:
        Initialized AutonomousAgent
    """
    return AutonomousAgent(
        llm=llm,
        perception=perception,
        reasoning=reasoning,
        memory=memory,
        execution=execution,
        **kwargs
    )
