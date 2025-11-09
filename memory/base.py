# memory/base.py
"""
Paper: Memory System
"A memory system that retains knowledge through both 
short-term and long-term mechanisms, enabling the agent 
to learn from past experiences and recall relevant information."
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from core.agent_context import AgentContext

@dataclass
class MemoryEntry:
    """Single memory entry"""
    timestamp: str
    perception: Dict
    action: Dict
    result: Any
    success: bool
    metadata: Dict = None

class MemorySystem(ABC):
    """
    Abstract Memory System
    
    Paper: Two-phase memory operation
    1. RETRIEVAL: Fetch relevant past experiences before reasoning
    2. STORAGE: Store current experience for future use
    """
    
    @abstractmethod
    def retrieve_context(self, 
                        query: str,
                        context: AgentContext) -> AgentContext:
        """
        Phase 1: RETRIEVAL
        
        Enrich context with similar past experiences
        Called BEFORE reasoning, so reasoning has context
        
        Paper: "Memory provides relevant past experiences 
        to inform reasoning process"
        """
        pass
    
    @abstractmethod
    def store_experience(self, context: AgentContext) -> None:
        """
        Phase 2: STORAGE
        
        Store current experience for future retrieval
        Called AFTER execution, for continuous learning
        
        Paper: "Learn from success/failure patterns"
        """
        pass
    
    @abstractmethod
    def extract_patterns(self, experiences: List[MemoryEntry]) -> Dict:
        """
        Extract learned patterns from experiences
        Paper: Pattern Learning capability
        """
        pass


class ShortTermMemory(ABC):
    """
    Paper: Short-term memory
    "Recent actions and decisions in FIFO manner"
    - Fixed size buffer (last N items)
    - Fast access
    - Provides immediate context
    """
    
    @abstractmethod
    def add(self, item: Dict) -> None:
        """Add to short-term memory"""
        pass
    
    @abstractmethod
    def get_recent(self, n: int = 3) -> List[Dict]:
        """Get last N items"""
        pass


class LongTermMemory(ABC):
    """
    Paper: Long-term memory
    "Semantic memory via vector database"
    - Stores all past experiences
    - Semantic search capability
    - Pattern recognition
    """
    
    @abstractmethod
    def store(self, experience: MemoryEntry) -> None:
        """Store experience in long-term memory"""
        pass
    
    @abstractmethod
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        """Retrieve similar past experiences"""
        pass
    
    @abstractmethod
    def extract_success_patterns(self) -> Dict:
        """What actions led to success?"""
        pass
    
    @abstractmethod
    def extract_failure_patterns(self) -> Dict:
        """What actions led to failure?"""
        pass
