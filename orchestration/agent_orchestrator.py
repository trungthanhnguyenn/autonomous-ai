# orchestration/agent_orchestrator.py
"""
Main Agent Orchestrator
========================

Coordinates all 4 systems according to autonomous agent architecture.

Purpose:
--------
- Automates the complete agent loop
- Handles communication between components
- Manages iteration and feedback
- Provides high-level abstraction for agent execution

Why use Orchestrator:
---------------------
1. Simplifies agent usage (1 line vs 5+ lines)
2. Ensures correct execution order
3. Handles errors gracefully
4. Provides consistent logging
5. Manages state transitions

Usage:
------
```python
orchestrator = AgentOrchestrator(
    perception=TextPerceptionSystem(),
    reasoning=LLMReasoningSystem(llm_client),
    memory=MemoryManager(),
    execution=ActionExecutor(tool_registry)
)

# Single step
context = orchestrator.execute_step(context, user_input)

# Multi-step with auto-feedback
result = orchestrator.run(user_input, max_iterations=5)
```
"""

from typing import Any, Dict, Optional
from core.agent_context import AgentContext, ExecutionStatus
from perception.base import PerceptionSystem
from reasoning.base import ReasoningSystem
from memory.base import MemorySystem
from execution.base import ExecutionSystem

class AgentOrchestrator:
    """
    Autonomous Agent Orchestrator
    
    Coordinates the complete agent loop following the paper architecture:
    
    Flow:
    -----
    1. PERCEPTION:     Understand input from environment
    2. MEMORY RETRIEVAL: Retrieve relevant past experiences (RAG)
    3. REASONING:      Decide what action to take (LLM)
    4. EXECUTION:      Execute the action (Tools)
    5. MEMORY STORAGE: Store experience for learning
    6. FEEDBACK:       Results inform next iteration
    
    Benefits:
    ---------
    - Single entry point for agent execution
    - Automatic error handling and logging
    - Consistent execution order
    - Built-in iteration management
    - Easy to monitor and debug
    """
    
    def __init__(self,
                 perception: PerceptionSystem,
                 reasoning: ReasoningSystem,
                 memory: MemorySystem,
                 execution: ExecutionSystem,
                 verbose: bool = True):
        """
        Initialize orchestrator with all 4 core systems
        
        Args:
            perception: System to process input
            reasoning: System to make decisions
            memory: System to store/retrieve experiences
            execution: System to execute actions
            verbose: Print progress messages
        """
        self.perception = perception
        self.reasoning = reasoning
        self.memory = memory
        self.execution = execution
        self.verbose = verbose
    
    def execute_step(self, 
                    context: AgentContext, 
                    env_input: Any,
                    stop_on_low_confidence: bool = False,
                    min_confidence: float = 0.3) -> AgentContext:
        """
        Execute single agent step with all 4 systems
        
        This is the core agent loop that processes one interaction.
        
        Args:
            context: Current agent context
            env_input: Input from environment (user query, sensor data, etc.)
            stop_on_low_confidence: Stop if reasoning confidence is too low
            min_confidence: Minimum confidence threshold
            
        Returns:
            Updated context after full loop
        """
        context.iteration += 1
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🔄 Iteration {context.iteration}")
            print('='*80)
        
        context.log_step("Orchestrator", f"Starting iteration {context.iteration}")
        
        try:
            # === STEP 1: PERCEPTION ===
            if self.verbose:
                print(f"\n[Iteration {context.iteration}] PERCEPTION")
            context = self.perception.perceive(env_input, context)
            
            # === STEP 2: MEMORY RETRIEVAL ===
            if self.verbose:
                print(f"[Iteration {context.iteration}] MEMORY RETRIEVAL")
            memory_query = self._build_memory_query(context)
            context = self.memory.retrieve_context(memory_query, context)
            
            # === STEP 3: REASONING ===
            if self.verbose:
                print(f"[Iteration {context.iteration}] REASONING")
            context = self.reasoning.reason(context)
            
            # Check confidence
            if stop_on_low_confidence and context.reasoning:
                if context.reasoning.confidence < min_confidence:
                    if self.verbose:
                        print(f"  ⚠️  Low confidence ({context.reasoning.confidence:.1%}), stopping")
                    context.log_step("Orchestrator", f"Stopped: Low confidence")
                    return context
            
            # === STEP 4: EXECUTION ===
            if self.verbose:
                print(f"[Iteration {context.iteration}] EXECUTION")
            context = self.execution.execute(context)
            
            # === STEP 5: MEMORY STORAGE ===
            if self.verbose:
                print(f"[Iteration {context.iteration}] MEMORY STORAGE")
            self.memory.store_experience(context)
            
            context.log_step("Orchestrator", f"Completed iteration {context.iteration}")
            
        except Exception as e:
            error_msg = f"Error in iteration {context.iteration}: {str(e)}"
            context.log_step("Orchestrator", error_msg)
            if self.verbose:
                print(f"  ❌ {error_msg}")
            raise
        
        return context
    
    def run(self,
            env_input: Any,
            max_iterations: int = 5,
            stop_on_completion: bool = True,
            stop_on_low_confidence: bool = True,
            context: Optional[AgentContext] = None) -> Dict:
        """
        Run complete agent loop with automatic iteration
        
        This is the high-level API for running the agent.
        
        Args:
            env_input: Initial input from environment
            max_iterations: Maximum number of iterations
            stop_on_completion: Stop when execution completes successfully
            stop_on_low_confidence: Stop if confidence drops too low
            context: Existing context (creates new if None)
            
        Returns:
            Dictionary with execution summary:
            {
                "success": bool,
                "total_iterations": int,
                "final_status": str,
                "final_result": Any,
                "context": AgentContext
            }
        """
        # Initialize context if needed
        if context is None:
            context = AgentContext()
        
        current_input = env_input
        final_status = "pending"
        
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 Starting Agent Execution")
            print("="*80)
            print(f"  Max iterations: {max_iterations}")
            print(f"  Stop on completion: {stop_on_completion}")
            print(f"  Stop on low confidence: {stop_on_low_confidence}")
        
        for iteration in range(max_iterations):
            # Execute one step
            context = self.execute_step(
                context, 
                current_input,
                stop_on_low_confidence=stop_on_low_confidence
            )
            
            # Check execution status
            if context.execution:
                final_status = context.execution.action_status.value
                
                if stop_on_completion and context.execution.action_status == ExecutionStatus.COMPLETED:
                    if self.verbose:
                        print(f"\n✅ Execution completed successfully")
                    break
                
                elif context.execution.action_status == ExecutionStatus.FAILED:
                    if self.verbose:
                        print(f"\n❌ Execution failed")
                    break
            
            # Check reasoning confidence
            if stop_on_low_confidence and context.reasoning:
                if context.reasoning.confidence < 0.3:
                    if self.verbose:
                        print(f"\n⚠️  Low confidence, stopping")
                    break
            
            # Prepare feedback for next iteration
            if context.execution and context.execution.result:
                current_input = {
                    "type": "feedback",
                    "previous_action": context.execution.action,
                    "result": context.execution.result,
                    "original_input": env_input
                }
        
        success = (context.execution and 
                  context.execution.action_status == ExecutionStatus.COMPLETED)
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"🏁 Agent Execution Complete")
            print("="*80)
            print(f"  Status: {'✅ Success' if success else '❌ Failed'}")
            print(f"  Iterations: {context.iteration}")
            print(f"  Final status: {final_status}")
            print("="*80 + "\n")
        
        return {
            "success": success,
            "total_iterations": context.iteration,
            "final_status": final_status,
            "final_result": context.execution.result if context.execution else None,
            "context": context
        }
    
    def execute_loop(self, 
                    context: AgentContext, 
                    env_input: Any, 
                    max_iterations: int = 5) -> Dict:
        """
        Legacy method - use run() instead
        
        Multi-step loop with feedback.
        Each iteration's output becomes next iteration's input.
        """
        return self.run(
            env_input=env_input,
            max_iterations=max_iterations,
            context=context
        )
    
    def _build_memory_query(self, context: AgentContext) -> str:
        """Build query for memory retrieval"""
        # Build meaningful query from perception
        intent = context.get_perceived_intent() or "unknown_intent"
        entities = [obj.get("value", "") for obj in context.get_extracted_entities()]
        
        # Combine intent + entities for semantic search
        query_parts = [intent] + entities
        return " ".join(query_parts)
