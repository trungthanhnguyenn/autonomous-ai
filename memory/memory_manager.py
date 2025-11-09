# memory/memory_manager.py
"""
Concrete Memory System Implementation
Combines short-term + long-term memory

Based on industry standards:
- OpenAI Assistants API: Thread-based conversation memory
- Anthropic Claude: Message history with context window management
- Vector-based semantic search for long-term memory
"""

import time
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
from core.agent_context import AgentContext, MemoryData, ExecutionStatus
from memory.base import (
    MemorySystem, 
    MemoryEntry, 
    ShortTermMemory,
    LongTermMemory
)

class SimpleShortTermMemory(ShortTermMemory):
    """
    FIFO buffer for recent interactions
    
    Inspired by OpenAI's Thread concept and Anthropic's message history.
    Maintains recent conversation context with:
    - Fixed size sliding window
    - Timestamp tracking
    - Metadata for context
    """
    
    def __init__(self, max_size: int = 10):
        """
        Args:
            max_size: Maximum number of recent items to keep (default: 10)
                     OpenAI typically uses 20-50 messages
                     Anthropic uses token-based limits (~100k tokens)
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, item: Dict) -> None:
        """
        Add item to short-term memory with timestamp
        
        Format (OpenAI-inspired):
        {
            "timestamp": ISO format,
            "role": "user" | "assistant" | "system",
            "data": actual content,
            "metadata": additional context
        }
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "data": item,
            "metadata": {
                "entry_id": len(self.buffer),
                "type": item.get("type", "interaction")
            }
        }
        self.buffer.append(entry)
    
    def get_recent(self, n: int = 3) -> List[Dict]:
        """
        Get last N items from memory
        
        Args:
            n: Number of recent items to retrieve
            
        Returns:
            List of recent memory entries
        """
        return list(self.buffer)[-n:] if n > 0 else []
    
    def get_all(self) -> List[Dict]:
        """Get all items in short-term memory"""
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear short-term memory (useful for new sessions)"""
        self.buffer.clear()
    
    def get_context_window(self, max_tokens: int = 4096) -> List[Dict]:
        """
        Get items that fit within a token budget
        
        Anthropic-inspired: Manage context window by token count
        Simple approximation: ~4 chars = 1 token
        """
        items = []
        estimated_tokens = 0
        
        for item in reversed(self.buffer):
            # Rough token estimation
            item_str = str(item)
            item_tokens = len(item_str) // 4
            
            if estimated_tokens + item_tokens > max_tokens:
                break
            
            items.insert(0, item)
            estimated_tokens += item_tokens
        
        return items


class SimpleLongTermMemory(LongTermMemory):
    """
    Long-term episodic memory with semantic search
    
    Inspired by:
    - OpenAI: Vector-based retrieval with embeddings
    - Anthropic: Structured knowledge base with metadata
    
    Features:
    - Stores all experiences permanently
    - Semantic similarity search (simulated with keyword matching for now)
    - Pattern extraction for learning
    - Success/failure analytics
    """
    
    def __init__(self):
        self.experiences: List[MemoryEntry] = []
        self.patterns_cache: Optional[Dict] = None
        self.last_pattern_update: Optional[datetime] = None
    
    def store(self, experience: MemoryEntry) -> None:
        """
        Store experience in long-term memory
        
        OpenAI-style: Each experience is like a message in a thread
        Anthropic-style: Includes metadata for filtering and retrieval
        """
        self.experiences.append(experience)
        # Invalidate patterns cache
        self.patterns_cache = None
    
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        """
        Retrieve similar experiences using semantic search
        
        Current: Simple keyword matching
        Production: Use vector embeddings (OpenAI text-embedding-ada-002)
        
        Args:
            query: Search query
            top_k: Number of similar experiences to return
            
        Returns:
            List of similar memory entries
        """
        if not self.experiences:
            return []
        
        # Score experiences by relevance
        scored_experiences = []
        query_lower = query.lower()
        
        for exp in self.experiences:
            score = 0.0
            
            # Check perception content
            perception_str = str(exp.perception).lower()
            if query_lower in perception_str:
                score += 3.0
            
            # Check action
            action_str = str(exp.action).lower()
            if query_lower in action_str:
                score += 2.0
            
            # Boost successful experiences
            if exp.success:
                score += 1.0
            
            # Recency bonus (more recent = slightly higher score)
            try:
                exp_time = datetime.fromisoformat(exp.timestamp)
                age_days = (datetime.now() - exp_time).days
                recency_score = max(0, 1.0 - (age_days / 30))  # Decay over 30 days
                score += recency_score * 0.5
            except:
                pass
            
            if score > 0:
                scored_experiences.append((score, exp))
        
        # Sort by score (descending) and return top_k
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored_experiences[:top_k]]
    
    def extract_success_patterns(self) -> Dict:
        """
        Analyze successful experiences to extract patterns
        
        OpenAI-style: Learn from what works
        Returns actionable insights for future reasoning
        """
        successful = [e for e in self.experiences if e.success]
        if not successful:
            return {
                "success_count": 0,
                "successful_actions": [],
                "success_rate": 0.0,
                "top_tools": []
            }
        
        # Analyze tool usage
        tool_usage = {}
        for exp in successful:
            tool_name = exp.action.get("tool", "unknown")
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        # Get top tools by usage
        top_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "successful_actions": [e.action for e in successful[-10:]],  # Last 10
            "success_count": len(successful),
            "success_rate": len(successful) / len(self.experiences) if self.experiences else 0,
            "top_tools": [{"tool": tool, "count": count} for tool, count in top_tools],
            "recent_successes": len([e for e in successful[-20:] if e in successful])
        }
    
    def extract_failure_patterns(self) -> Dict:
        """
        Analyze failed experiences to learn what to avoid
        
        Anthropic-style: Learn from mistakes
        """
        failed = [e for e in self.experiences if not e.success]
        if not failed:
            return {
                "failure_count": 0,
                "failed_actions": [],
                "failure_rate": 0.0,
                "error_types": []
            }
        
        # Analyze tool failures
        tool_failures = {}
        for exp in failed:
            tool_name = exp.action.get("tool", "unknown")
            tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1
        
        top_failures = sorted(tool_failures.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "failed_actions": [e.action for e in failed[-10:]],
            "failure_count": len(failed),
            "failure_rate": len(failed) / len(self.experiences) if self.experiences else 0,
            "common_failures": [{"tool": tool, "count": count} for tool, count in top_failures]
        }
    
    def get_statistics(self) -> Dict:
        """Get overall memory statistics"""
        return {
            "total_experiences": len(self.experiences),
            "successful": len([e for e in self.experiences if e.success]),
            "failed": len([e for e in self.experiences if not e.success]),
            "oldest_entry": self.experiences[0].timestamp if self.experiences else None,
            "newest_entry": self.experiences[-1].timestamp if self.experiences else None
        }


class MemoryManager(MemorySystem):
    """
    Main Memory Manager - Orchestrates short-term and long-term memory
    
    Architecture inspired by:
    - OpenAI Assistants: Thread + Message model
    - Anthropic Claude: Context window management
    - LangChain: Memory chains and retrieval
    
    Paper-Aligned Flow:
    1. Before Reasoning: Retrieve similar experiences (RAG pattern)
    2. After Execution: Store new experience & update patterns
    3. Continuous Learning: Extract patterns from history
    """
    
    def __init__(self, 
                 short_term_size: int = 10,
                 enable_pattern_cache: bool = True):
        """
        Initialize memory manager
        
        Args:
            short_term_size: Size of short-term memory buffer
            enable_pattern_cache: Cache patterns for performance
        """
        self.short_term = SimpleShortTermMemory(max_size=short_term_size)
        self.long_term = SimpleLongTermMemory()
        self.patterns = {}
        self.enable_pattern_cache = enable_pattern_cache
        
        # Statistics
        self.retrieval_count = 0
        self.storage_count = 0
    
    def retrieve_context(self, 
                        query: str,
                        context: AgentContext) -> AgentContext:
        """
        Phase 1: RETRIEVAL (called before reasoning)
        
        Implements RAG (Retrieval-Augmented Generation) pattern:
        1. Search long-term memory for similar experiences
        2. Extract relevant patterns
        3. Combine with short-term context
        4. Enrich agent context for better reasoning
        
        OpenAI-style: Like searching through a thread's messages
        Anthropic-style: Context assembly within token limits
        """
        start_time = time.time()
        self.retrieval_count += 1
        
        # 1. Get similar experiences from long-term memory (semantic search)
        similar_experiences = self.long_term.retrieve_similar(query, top_k=3)
        
        # 2. Extract learned patterns (what worked/failed before)
        if self.enable_pattern_cache and self.patterns:
            patterns = self.patterns
        else:
            patterns = {
                "success_patterns": self.long_term.extract_success_patterns(),
                "failure_patterns": self.long_term.extract_failure_patterns()
            }
            if self.enable_pattern_cache:
                self.patterns = patterns
        
        # 3. Get recent short-term context
        recent_context = self.short_term.get_recent(n=5)
        
        # 4. Assemble memory data for agent context
        context.memory = MemoryData(
            similar_experiences=[self._format_experience(e) for e in similar_experiences],
            learned_patterns=patterns,
            recent_actions=recent_context,
            retrieval_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "statistics": self.long_term.get_statistics(),
                "query": query
            }
        )
        
        context.log_step(
            "Memory", 
            f"Retrieved {len(similar_experiences)} similar experiences, "
            f"{len(recent_context)} recent interactions"
        )
        
        return context
    
    def store_experience(self, context: AgentContext) -> None:
        """
        Phase 2: STORAGE (called after execution)
        
        Store experience for continuous learning:
        1. Extract key information from context
        2. Store in short-term memory (immediate recall)
        3. Store in long-term memory (semantic search)
        4. Update patterns cache
        
        OpenAI-style: Like adding a message to a thread
        Anthropic-style: Building knowledge base over time
        """
        self.storage_count += 1
        
        # Validate context has required data
        if not context.perception or not context.execution:
            context.log_step("Memory", "⚠ Incomplete context, skipping storage")
            return
        
        # Create memory entry
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            perception=context.perception.to_dict(),
            action=context.execution.action or {},
            result=context.execution.result,
            success=context.execution.action_status == ExecutionStatus.COMPLETED,
            metadata={
                "iteration": context.iteration,
                "confidence": getattr(context.reasoning, 'confidence', 0.0) if context.reasoning else 0.0,
                "execution_time": getattr(context.execution, 'execution_time_ms', 0.0)
            }
        )
        
        # Store in both memories
        self.short_term.add(entry.__dict__)
        self.long_term.store(entry)
        
        # Invalidate patterns cache for fresh data
        if self.enable_pattern_cache:
            self.patterns = {}
        
        # Update patterns periodically (every 10 experiences)
        if self.storage_count % 10 == 0:
            self._update_patterns()
        
        context.log_step(
            "Memory",
            f"Stored experience (success={entry.success}). "
            f"Total: {len(self.long_term.experiences)} experiences"
        )
    
    def extract_patterns(self, experiences: List[MemoryEntry]) -> Dict:
        """
        Extract learning patterns from experiences
        
        Returns insights that reasoning system can use:
        - What actions work well
        - What to avoid
        - Common failure modes
        """
        if not experiences:
            return {
                "success_patterns": {},
                "failure_patterns": {},
                "insights": []
            }
        
        success_patterns = self.long_term.extract_success_patterns()
        failure_patterns = self.long_term.extract_failure_patterns()
        
        # Generate actionable insights
        insights = self._generate_insights(success_patterns, failure_patterns)
        
        return {
            "success_patterns": success_patterns,
            "failure_patterns": failure_patterns,
            "insights": insights,
            "total_experiences": len(experiences)
        }
    
    def _update_patterns(self):
        """Update internal patterns cache"""
        self.patterns = self.extract_patterns(self.long_term.experiences)
    
    def _format_experience(self, entry: MemoryEntry) -> Dict:
        """Format memory entry for context"""
        return {
            "timestamp": entry.timestamp,
            "perception": entry.perception,
            "action": entry.action,
            "result": str(entry.result)[:200] if entry.result else None,  # Truncate long results
            "success": entry.success,
            "metadata": entry.metadata
        }
    
    def _generate_insights(self, 
                          success_patterns: Dict, 
                          failure_patterns: Dict) -> List[str]:
        """
        Generate actionable insights from patterns
        
        Anthropic-style: Convert data into natural language insights
        """
        insights = []
        
        # Success insights
        if success_patterns.get("top_tools"):
            top_tool = success_patterns["top_tools"][0]
            insights.append(
                f"Most successful tool: '{top_tool['tool']}' "
                f"(used {top_tool['count']} times successfully)"
            )
        
        success_rate = success_patterns.get("success_rate", 0)
        if success_rate > 0:
            insights.append(f"Overall success rate: {success_rate:.1%}")
        
        # Failure insights
        if failure_patterns.get("common_failures"):
            top_failure = failure_patterns["common_failures"][0]
            insights.append(
                f"Common failure: '{top_failure['tool']}' "
                f"(failed {top_failure['count']} times)"
            )
        
        return insights
    
    def get_memory_summary(self) -> Dict:
        """
        Get comprehensive memory summary
        
        Useful for debugging and monitoring
        """
        return {
            "short_term": {
                "size": len(self.short_term.buffer),
                "max_size": self.short_term.max_size,
                "items": self.short_term.get_all()
            },
            "long_term": self.long_term.get_statistics(),
            "patterns": self.patterns,
            "operations": {
                "retrievals": self.retrieval_count,
                "storages": self.storage_count
            }
        }
    
    def clear_short_term(self):
        """Clear short-term memory (e.g., new conversation)"""
        self.short_term.clear()
    
    def reset(self):
        """Reset all memory (use carefully!)"""
        self.short_term.clear()
        self.long_term.experiences.clear()
        self.patterns = {}
        self.retrieval_count = 0
        self.storage_count = 0
