# core/agent_context.py
"""
Agent Context - Shared State Container

This is the central data structure that flows through all 4 systems.
Each system reads from and writes to this context.

Paper Concept:
"A unified representation of the agent's understanding of its state,
goals, environment, and past experiences"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import json
import uuid


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class PerceptionModality(Enum):
    """
    Paper Section 3: Different perception input types
    Agents can perceive through multiple modalities
    """
    TEXT = "text"                      # Natural language input
    IMAGE = "image"                    # Visual input
    VIDEO = "video"                    # Video stream
    AUDIO = "audio"                    # Audio input
    STRUCTURED = "structured"          # JSON, HTML, A11y trees
    SENSOR = "sensor"                  # Real-world sensors
    TOOL = "tool"                      # Tool/API outputs


class ReasoningMethod(Enum):
    """
    Paper Section 4: Different reasoning approaches
    Reasoning system can use different strategies
    """
    CHAIN_OF_THOUGHT = "cot"           # Step-by-step thinking
    TREE_OF_THOUGHT = "tot"            # Multi-path exploration
    ADAPTIVE = "adaptive"              # Learning from feedback
    REFLECTION = "reflection"          # Self-evaluation


class ExecutionStatus(Enum):
    """Status of action execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ============================================================================
# DATA CLASSES FOR EACH SYSTEM
# ============================================================================

@dataclass
class PerceptualData:
    """
    Output of Perception System
    
    Paper: "Perception converts environmental percepts 
    into meaningful representations"
    
    Contains everything perception extracted from input
    """
    # Input information
    modality: PerceptionModality
    raw_input: Any                          # Original unprocessed input
    
    # Extracted information
    extracted_objects: List[Dict] = field(default_factory=list)
    """
    Recognized entities/objects in the environment
    Format: [{"type": "EMAIL", "value": "user@example.com", "confidence": 0.95}, ...]
    """
    
    semantic_representation: Dict = field(default_factory=dict)
    """
    Meaningful interpretation of environment
    Format: {"intent": "search", "keywords": ["python", "tutorial"], "complexity": "moderate"}
    """
    
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    """
    Confidence for each extracted component
    Format: {"entity_recognition": 0.95, "intent_detection": 0.85, "overall": 0.90}
    """
    
    # Metadata
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    """Additional perception metadata"""
    
    def to_dict(self) -> Dict:
        """Convert to dict for serialization"""
        return {
            "modality": self.modality.value,
            "raw_input": str(self.raw_input)[:200],  # Truncate for readability
            "extracted_objects": self.extracted_objects,
            "semantic_representation": self.semantic_representation,
            "confidence_scores": self.confidence_scores,
            "processing_time_ms": self.processing_time_ms
        }
    
    def get_confidence(self, component: str = "overall") -> float:
        """Get confidence score for specific component"""
        return self.confidence_scores.get(component, 0.0)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if perception is confident enough"""
        return self.get_confidence("overall") >= threshold


@dataclass
class ReasoningData:
    """
    Output of Reasoning System
    
    Paper: "Reasoning formulates plans, adapts to feedback,
    and evaluates actions through Chain-of-Thought and Tree-of-Thought"
    
    Contains reasoning process and generated action plan
    """
    # Goal and planning
    current_goal: Optional[str] = None
    """What the agent is trying to achieve"""
    
    action_plan: List[Dict] = field(default_factory=list)
    """
    Sequence of actions to execute
    Format: [
        {"tool": "search", "params": {"query": "..."}},
        {"tool": "navigate", "params": {"url": "..."}}
    ]
    """
    
    # Reasoning process
    reasoning_method: ReasoningMethod = ReasoningMethod.CHAIN_OF_THOUGHT
    reasoning_trace: List[str] = field(default_factory=list)
    """
    Step-by-step thinking process (Chain-of-Thought trace)
    Shows how LLM arrived at the plan
    """
    
    alternative_plans: List[List[Dict]] = field(default_factory=list)
    """
    Alternative action plans (from Tree-of-Thought)
    For selection when primary plan fails
    """
    
    # Confidence and quality metrics
    confidence: float = 0.0
    """
    Overall confidence in the plan (0.0 - 1.0)
    Used to decide: proceed or request alternative
    """
    
    plan_quality_metrics: Dict[str, float] = field(default_factory=dict)
    """
    Quality metrics for the plan
    Format: {"clarity": 0.9, "feasibility": 0.8, "efficiency": 0.75}
    """
    
    # Metadata
    reasoning_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def get_next_action(self) -> Optional[Dict]:
        """Get first pending action from plan"""
        return self.action_plan[0] if self.action_plan else None
    
    def has_alternatives(self) -> bool:
        """Check if alternatives exist (ToT)"""
        return len(self.alternative_plans) > 0
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if reasoning confidence is sufficient"""
        return self.confidence >= threshold


@dataclass
class ExecutionData:
    """
    Output of Execution System
    
    Paper: "Execution translates internal decisions 
    into concrete actions"
    
    Contains action execution details and results
    """
    # Current action being executed
    action: Optional[Dict] = None
    """
    The action being executed
    Format: {"tool": "search", "params": {"query": "..."}}
    """
    
    action_status: ExecutionStatus = ExecutionStatus.PENDING
    """Current status of action execution"""
    
    # Execution results
    result: Optional[Any] = None
    """Result/output from executing the action"""
    
    error: Optional[str] = None
    """Error message if execution failed"""
    
    # Feedback information
    feedback: Optional[str] = None
    """Feedback from environment for learning"""
    
    # Metadata
    execution_time_ms: float = 0.0
    retries: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def was_successful(self) -> bool:
        """Check if action succeeded"""
        return self.action_status == ExecutionStatus.COMPLETED
    
    def get_feedback_summary(self) -> str:
        """Get human-readable feedback"""
        if self.was_successful():
            return f"✓ Action succeeded: {self.result}"
        else:
            return f"✗ Action failed: {self.error}"


@dataclass
class MemoryData:
    """
    Context data from Memory System
    
    Paper: "Memory retains knowledge through short-term 
    and long-term mechanisms"
    
    Contains retrieved experiences and learned patterns
    """
    # Retrieved experiences
    similar_experiences: List[Dict] = field(default_factory=list)
    """
    Similar past experiences retrieved from long-term memory
    Format: [
        {"action": {...}, "result": {...}, "success": True},
        ...
    ]
    """
    
    # Learned patterns
    learned_patterns: Dict = field(default_factory=dict)
    """
    Extracted patterns from past experiences
    Format: {
        "success_patterns": {...},
        "failure_patterns": {...},
        "common_sequences": [...]
    }
    """
    
    # Short-term context
    recent_actions: List[Dict] = field(default_factory=list)
    """Recent actions (FIFO buffer)"""
    
    # Metadata
    retrieval_time_ms: float = 0.0
    pattern_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def has_similar_experiences(self) -> bool:
        """Check if similar experiences were found"""
        return len(self.similar_experiences) > 0
    
    def has_patterns(self) -> bool:
        """Check if patterns were learned"""
        return len(self.learned_patterns) > 0


@dataclass
class ExecutionLog:
    """Single log entry for system execution"""
    timestamp: datetime
    system: str              # "Perception", "Reasoning", "Memory", "Execution"
    level: str              # "INFO", "WARNING", "ERROR", "DEBUG"
    message: str
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        return f"[{time_str}] {self.system:12} | {self.level:7} | {self.message}"


# ============================================================================
# MAIN AGENT CONTEXT
# ============================================================================

@dataclass
class AgentContext:
    """
    Central Agent Context
    
    This is the unified data structure that flows through all 4 systems:
    Perception → Memory → Reasoning → Execution
    
    Each system reads from context, processes, and writes back to context.
    
    Paper Concept:
    "A comprehensive representation of the agent's state, including
    its perception of the environment, reasoning about actions,
    memory of past experiences, and execution status"
    """
    
    # === IDENTIFIERS ===
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique session identifier"""
    
    iteration: int = 0
    """Current iteration number"""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """Context creation timestamp"""
    
    # === DATA FROM EACH SYSTEM ===
    perception: PerceptualData = field(default_factory=lambda: PerceptualData(
        modality=PerceptionModality.TEXT,
        raw_input=None
    ))
    """Data extracted by perception system"""
    
    memory: MemoryData = field(default_factory=MemoryData)
    """Context enriched by memory system"""
    
    reasoning: ReasoningData = field(default_factory=ReasoningData)
    """Plan generated by reasoning system"""
    
    execution: ExecutionData = field(default_factory=ExecutionData)
    """Result from execution system"""
    
    # === EXECUTION LOGS ===
    logs: List[ExecutionLog] = field(default_factory=list)
    """Execution trace for debugging"""
    
    # === HISTORY ===
    history: List[Dict] = field(default_factory=list)
    """History of past contexts (for learning)"""
    
    # === GENERAL METADATA ===
    metadata: Dict = field(default_factory=dict)
    """
    General purpose metadata
    Can store system-specific data
    """
    
    # === STATUS ===
    is_complete: bool = False
    """Whether goal is achieved"""
    
    completion_reason: Optional[str] = None
    """Why the agent stopped (success, max_iterations, error)"""
    
    # ========================================================================
    # METHODS - LOGGING
    # ========================================================================
    
    def log_step(self, system_name: str, message: str, level: str = "INFO", **metadata):
        """
        Log a step in the agent execution
        
        Args:
            system_name: Which system (Perception, Reasoning, Memory, Execution)
            message: What happened
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            **metadata: Additional metadata
        """
        log_entry = ExecutionLog(
            timestamp=datetime.now(),
            system=system_name,
            level=level,
            message=message,
            metadata=metadata
        )
        self.logs.append(log_entry)
        
        # Also print for visibility
        if level in ["WARNING", "ERROR"]:
            print(f"⚠️  {log_entry}")
    
    def get_execution_trace(self) -> str:
        """Get formatted execution trace"""
        return "\n".join(str(log) for log in self.logs)
    
    # ========================================================================
    # METHODS - PERCEPTION
    # ========================================================================
    
    def has_perception(self) -> bool:
        """Check if perception data exists"""
        return self.perception.raw_input is not None
    
    def get_perceived_intent(self) -> Optional[str]:
        """Get perceived user intent"""
        return self.perception.semantic_representation.get("intent")
    
    def get_extracted_entities(self) -> List[Dict]:
        """Get recognized entities"""
        return self.perception.extracted_objects
    
    # ========================================================================
    # METHODS - REASONING
    # ========================================================================
    
    def has_action_plan(self) -> bool:
        """Check if action plan exists"""
        return len(self.reasoning.action_plan) > 0
    
    def get_next_action(self) -> Optional[Dict]:
        """Get next action to execute"""
        return self.reasoning.get_next_action()
    
    def get_reasoning_confidence(self) -> float:
        """Get reasoning confidence"""
        return self.reasoning.confidence
    
    def get_reasoning_trace(self) -> List[str]:
        """Get step-by-step reasoning"""
        return self.reasoning.reasoning_trace
    
    # ========================================================================
    # METHODS - MEMORY
    # ========================================================================
    
    def has_memory_context(self) -> bool:
        """Check if memory enriched context"""
        return (self.memory.has_similar_experiences() or 
                self.memory.has_patterns())
    
    def get_similar_experiences(self) -> List[Dict]:
        """Get similar past experiences"""
        return self.memory.similar_experiences
    
    def get_learned_patterns(self) -> Dict:
        """Get learned patterns"""
        return self.memory.learned_patterns
    
    # ========================================================================
    # METHODS - EXECUTION
    # ========================================================================
    
    def is_action_completed(self) -> bool:
        """Check if action execution completed"""
        return self.execution.was_successful()
    
    def get_execution_result(self) -> Optional[Any]:
        """Get result from last execution"""
        return self.execution.result
    
    def get_execution_error(self) -> Optional[str]:
        """Get error if execution failed"""
        return self.execution.error
    
    # ========================================================================
    # METHODS - CONTEXT STATE
    # ========================================================================
    
    def get_state_summary(self) -> Dict:
        """Get summary of current context state"""
        return {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "perception_modality": self.perception.modality.value,
            "perception_confidence": self.perception.get_confidence(),
            "has_action_plan": self.has_action_plan(),
            "reasoning_confidence": self.get_reasoning_confidence(),
            "has_memory_context": self.has_memory_context(),
            "action_status": self.execution.action_status.value if self.execution.action_status else None,
            "is_complete": self.is_complete,
        }
    
    def to_dict(self, include_logs: bool = False) -> Dict:
        """
        Convert context to dict for serialization
        
        Args:
            include_logs: Include detailed logs (can be large)
        """
        result = {
            "session_id": self.session_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "perception": self.perception.to_dict(),
            "memory": {
                "similar_experiences_count": len(self.memory.similar_experiences),
                "learned_patterns_count": len(self.memory.learned_patterns),
            },
            "reasoning": {
                "goal": self.reasoning.current_goal,
                "action_plan_count": len(self.reasoning.action_plan),
                "confidence": self.reasoning.confidence,
                "reasoning_method": self.reasoning.reasoning_method.value,
            },
            "execution": {
                "action_status": self.execution.action_status.value,
                "was_successful": self.execution.was_successful(),
                "retries": self.execution.retries,
            },
            "completion": {
                "is_complete": self.is_complete,
                "reason": self.completion_reason,
            }
        }
        
        if include_logs:
            result["logs"] = [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "system": log.system,
                    "level": log.level,
                    "message": log.message,
                }
                for log in self.logs
            ]
        
        return result
    
    def to_json(self, include_logs: bool = False, indent: int = 2) -> str:
        """Convert context to JSON string"""
        return json.dumps(self.to_dict(include_logs=include_logs), indent=indent)
    
    # ========================================================================
    # METHODS - HISTORY & LEARNING
    # ========================================================================
    
    def save_to_history(self) -> None:
        """Save current context snapshot to history"""
        snapshot = self.to_dict(include_logs=False)
        self.history.append(snapshot)
    
    def get_history_length(self) -> int:
        """Get number of history snapshots"""
        return len(self.history)
    
    def can_learn_from_history(self, min_examples: int = 3) -> bool:
        """Check if enough history for learning"""
        return len(self.history) >= min_examples
    
    # ========================================================================
    # METHODS - VALIDATION & CHECKS
    # ========================================================================
    
    def validate_flow(self) -> Tuple[bool, List[str]]:
        """
        Validate that context has valid flow
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check perception
        if not self.has_perception():
            errors.append("No perception data")
        
        # Check reasoning after memory
        if self.has_memory_context() and not self.has_action_plan():
            errors.append("Memory context but no action plan")
        
        # Check execution only if plan exists
        if self.has_action_plan() and self.execution.action is None:
            errors.append("Action plan but no execution attempt")
        
        return len(errors) == 0, errors
    
    def reset(self) -> None:
        """Reset context for new iteration (keep history)"""
        self.perception = PerceptualData(modality=PerceptionModality.TEXT, raw_input=None)
        self.memory = MemoryData()
        self.reasoning = ReasoningData()
        self.execution = ExecutionData()
        self.logs = []
    
    # ========================================================================
    # METHODS - DEBUG & VISUALIZATION
    # ========================================================================
    
    def pretty_print(self) -> str:
        """
        Pretty print context state with detailed information
        Useful for debugging and monitoring agent flow
        """
        output = []
        
        # Header
        output.append("\n" + "═" * 80)
        output.append(f"║  🤖 AGENT CONTEXT - Iteration {self.iteration}".ljust(79) + "║")
        output.append("═" * 80)
        
        # === PERCEPTION ===
        output.append("\n📥 PERCEPTION")
        output.append("─" * 80)
        if self.perception:
            output.append(f"  Modality:      {self.perception.modality.value.upper()}")
            output.append(f"  Raw Input:     {str(self.perception.raw_input)[:60]}...")
            
            # Semantic representation
            semantic = self.perception.semantic_representation
            if semantic:
                output.append(f"  Intent:        {semantic.get('intent', 'N/A')}")
                keywords = semantic.get('keywords', [])
                if keywords:
                    output.append(f"  Keywords:      {', '.join(keywords[:5])}")
                output.append(f"  Complexity:    {semantic.get('complexity', 'N/A')}")
            
            # Objects
            objects = self.perception.extracted_objects
            if objects:
                output.append(f"  Objects:       {len(objects)} detected")
                for i, obj in enumerate(objects[:3], 1):
                    output.append(f"                 {i}. {obj.get('type', 'N/A')}: {obj.get('value', 'N/A')}")
            
            output.append(f"  Confidence:    {self.perception.get_confidence():.1%}")
            output.append(f"  Process Time:  {self.perception.processing_time_ms:.2f}ms")
        else:
            output.append("  ⚠️  No perception data")
        
        # === MEMORY ===
        output.append("\n💾 MEMORY")
        output.append("─" * 80)
        if self.memory:
            similar = self.memory.similar_experiences
            output.append(f"  Similar Exp:   {len(similar)} retrieved")
            if similar:
                for i, exp in enumerate(similar[:2], 1):
                    action = exp.get('action', {})
                    success = exp.get('success', False)
                    status = "✓" if success else "✗"
                    output.append(f"                 {i}. {status} {action.get('tool', 'N/A')}")
            
            patterns = self.memory.learned_patterns
            if patterns:
                success_patterns = patterns.get('success_patterns', {})
                if success_patterns:
                    rate = success_patterns.get('success_rate', 0)
                    count = success_patterns.get('success_count', 0)
                    output.append(f"  Patterns:      {count} successful ({rate:.1%} success rate)")
                    
                    top_tools = success_patterns.get('top_tools', [])
                    if top_tools:
                        output.append(f"  Top Tools:     {', '.join([t['tool'] for t in top_tools[:3]])}")
            
            recent = self.memory.recent_actions
            output.append(f"  Recent Acts:   {len(recent)} in buffer")
            output.append(f"  Retrieval:     {self.memory.retrieval_time_ms:.2f}ms")
        else:
            output.append("  ⚠️  No memory data")
        
        # === REASONING ===
        output.append("\n🧠 REASONING")
        output.append("─" * 80)
        if self.reasoning:
            output.append(f"  Method:        {self.reasoning.reasoning_method.value}")
            
            # Goal
            goal = self.reasoning.current_goal or "Not set"
            output.append(f"  Goal:          {goal}")
            
            # Action plan
            plan = self.reasoning.action_plan
            output.append(f"  Action Plan:   {len(plan)} step(s)")
            for i, action in enumerate(plan[:3], 1):
                tool = action.get('tool', 'N/A')
                params = action.get('params', {})
                param_str = ', '.join([f"{k}={v}" for k, v in list(params.items())[:2]])
                if len(param_str) > 40:
                    param_str = param_str[:37] + "..."
                output.append(f"                 {i}. {tool}({param_str})")
            
            # Reasoning trace
            trace = self.reasoning.reasoning_trace
            if trace:
                output.append(f"  Thinking:      {trace[0][:60]}...")
            
            # Confidence
            conf = self.reasoning.confidence
            conf_emoji = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
            output.append(f"  Confidence:    {conf_emoji} {conf:.1%}")
            
            if self.reasoning.reasoning_time_ms:
                output.append(f"  Process Time:  {self.reasoning.reasoning_time_ms:.2f}ms")
        else:
            output.append("  ⚠️  No reasoning data")
        
        # === EXECUTION ===
        output.append("\n⚙️  EXECUTION")
        output.append("─" * 80)
        if self.execution:
            status = self.execution.action_status
            status_emoji = {
                ExecutionStatus.PENDING: "⏳",
                ExecutionStatus.EXECUTING: "🔄",
                ExecutionStatus.COMPLETED: "✅",
                ExecutionStatus.FAILED: "❌",
                ExecutionStatus.PARTIAL: "⚠️"
            }.get(status, "❓")
            
            output.append(f"  Status:        {status_emoji} {status.value.upper()}")
            
            # Action
            action = self.execution.action
            if action:
                tool = action.get('tool', 'N/A')
                output.append(f"  Tool:          {tool}")
            
            # Result
            result = self.execution.result
            if result:
                result_str = str(result)
                if len(result_str) > 60:
                    result_str = result_str[:57] + "..."
                output.append(f"  Result:        {result_str}")
            
            # Error
            error = self.execution.error
            if error:
                output.append(f"  ❌ Error:      {error}")
            
            if self.execution.execution_time_ms:
                output.append(f"  Exec Time:     {self.execution.execution_time_ms:.2f}ms")
        else:
            output.append("  ⚠️  No execution data")
        
        # === EXECUTION LOG ===
        if hasattr(self, 'execution_log') and self.execution_log:
            output.append("\n📋 EXECUTION LOG")
            output.append("─" * 80)
            for entry in self.execution_log[-5:]:  # Last 5 entries
                phase = entry.get('phase', 'N/A')
                msg = entry.get('message', 'N/A')
                output.append(f"  [{phase}] {msg}")
        
        # Footer
        output.append("\n" + "═" * 80 + "\n")
        
        return "\n".join(output)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_initial_context(input_data: Any, 
                          modality: PerceptionModality = PerceptionModality.TEXT) -> AgentContext:
    """
    Create initial context from raw input
    
    Args:
        input_data: Raw input from environment
        modality: How to interpret input
    
    Returns:
        AgentContext ready for processing
    """
    context = AgentContext()
    context.perception.raw_input = input_data
    context.perception.modality = modality
    return context


def merge_contexts(contexts: List[AgentContext]) -> AgentContext:
    """
    Merge multiple context snapshots
    Useful for multi-agent scenarios
    """
    merged = AgentContext()
    
    # Combine perceptions
    all_objects = []
    for ctx in contexts:
        all_objects.extend(ctx.perception.extracted_objects)
    merged.perception.extracted_objects = all_objects
    
    # Combine memories
    all_experiences = []
    for ctx in contexts:
        all_experiences.extend(ctx.memory.similar_experiences)
    merged.memory.similar_experiences = all_experiences
    
    return merged
