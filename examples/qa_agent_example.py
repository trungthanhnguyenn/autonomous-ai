#!/usr/bin/env python3
"""
Example: Q&A Agent with Google Search
======================================

This example demonstrates how to build a complete autonomous agent that can:
1. Receive user questions
2. Search Google for answers
3. Learn from past experiences
4. Improve over time

Architecture:
- Perception: Text input processing
- Memory: Short-term + long-term with patterns
- Reasoning: LLM-based decision making
- Execution: Google Search tool
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_context import AgentContext, PerceptionModality
from perception.text.text_perception import TextPerceptionSystem
from memory.memory_manager import MemoryManager
from reasoning.llm_reasoner import LLMReasoningSystem
from execution.action_executor import ActionExecutor
from execution.tools.tool_registry import ToolRegistry
from execution.tools.google_search_tool import GoogleSearchTool
from execution.tools.builtin_tools import SearchTool
from utils.llm_client import LLMClient


class QAAgent:
    """
    Q&A Agent with Google Search capabilities
    
    This agent follows the autonomous agent pattern:
    1. PERCEPTION: Understand user question
    2. MEMORY: Retrieve similar past Q&As
    3. REASONING: Decide what action to take
    4. EXECUTION: Execute action (search, respond, etc.)
    5. LEARNING: Store experience for future use
    """
    
    def __init__(self, use_real_google_search: bool = True):
        """
        Initialize Q&A Agent
        
        Args:
            use_real_google_search: Use real Google API or simulated search
        """
        print("\n" + "="*80)
        print("Initializing Q&A Agent...")
        print("="*80)
        
        # 1. Initialize Perception System
        print("\nSetting up Perception...")
        self.perception_system = TextPerceptionSystem()
        print("   ✓ Text perception ready")
        
        # 2. Initialize Memory System
        print("\nSetting up Memory...")
        self.memory_manager = MemoryManager(
            short_term_size=10,  # Keep last 10 interactions
            enable_pattern_cache=True  # Cache patterns for speed
        )
        print("Memory system ready (short-term: 10, long-term: unlimited)")
        
        # 3. Initialize Reasoning System (LLM)
        print("\nSetting up Reasoning...")
        try:
            self.llm_client = LLMClient()
            self.reasoning_system = LLMReasoningSystem(
                self.llm_client,
                tool_registry=None  # Will set after creating tools
            )
            print("LLM reasoning ready")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            print("   Agent will work with limited reasoning capabilities")
            self.reasoning_system = None
        
        # 4. Initialize Execution System with Tools
        print("\nSetting up Execution...")
        self.tool_registry = ToolRegistry()
        
        # Register Google Search tool
        if use_real_google_search:
            google_tool = GoogleSearchTool()
            self.tool_registry.register(google_tool)
            print("   ✓ Registered: google_search (Real Google API)")
        else:
            # Fallback to simulated search
            search_tool = SearchTool()
            self.tool_registry.register(search_tool)
            print("Registered: search (Simulated)")
        
        self.action_executor = ActionExecutor(self.tool_registry)
        
        # Link tool registry to reasoning
        if self.reasoning_system:
            self.reasoning_system.tool_registry = self.tool_registry
        
        print("Execution system ready")
        
        print("\n" + "="*80)
        print("Q&A Agent initialized successfully!")
        print("="*80 + "\n")
    
    def ask(self, question: str, verbose: bool = True) -> dict:
        """
        Process a question through the autonomous agent loop
        
        Args:
            question: User's question
            verbose: Print detailed flow information
            
        Returns:
            dict with answer and metadata
        """
        if verbose:
            print("\n" + "🔵"*40)
            print(f"Question: {question}")
            print("🔵"*40 + "\n")
        
        # Create initial context
        context = AgentContext()
        
        # === STEP 1: PERCEPTION ===
        if verbose:
            print("STEP 1: PERCEPTION")
            print("─"*80)
        
        context = self.perception_system.perceive(question, context)
        
        if verbose:
            print(f"Intent detected: {context.perception.semantic_representation.get('intent', 'N/A')}")
            print(f"Keywords: {', '.join(context.perception.semantic_representation.get('keywords', [])[:5])}")
            print(f"Confidence: {context.perception.get_confidence():.1%}\n")
        
        # === STEP 2: MEMORY RETRIEVAL ===
        if verbose:
            print("STEP 2: MEMORY RETRIEVAL")
            print("─"*80)
        
        context = self.memory_manager.retrieve_context(question, context)
        
        if verbose:
            similar = len(context.memory.similar_experiences)
            print(f"Retrieved {similar} similar past experiences")
            if similar > 0:
                print(f"Learning from past: ✓")
            print()
        
        # === STEP 3: REASONING ===
        if verbose:
            print("STEP 3: REASONING")
            print("─"*80)
        
        if self.reasoning_system:
            context = self.reasoning_system.reason(context)
            
            if verbose:
                action_plan = context.reasoning.action_plan
                if action_plan:
                    action = action_plan[0]
                    print(f"Decided action: {action.get('tool', 'N/A')}")
                    print(f"Parameters: {action.get('params', {})}")
                    print(f"Confidence: {context.reasoning.confidence:.1%}")
                print()
        else:
            # Fallback: Create default search action
            context.reasoning.action_plan = [{
                "tool": "google_search" if "google_search" in [t.name for t in self.tool_registry.tools.values()] else "search",
                "params": {"query": question}
            }]
            if verbose:
                print("   Using fallback reasoning (no LLM)")
                print()
        
        # === STEP 4: EXECUTION ===
        if verbose:
            print("STEP 4: EXECUTION")
            print("─"*80)
        
        context = self.action_executor.execute(context)
        
        if verbose:
            if context.execution.action_status.value == "completed":
                print(f"Status: Completed")
                result = context.execution.result
                if isinstance(result, dict) and "results" in result:
                    results = result["results"]
                    print(f"Found {len(results)} results")
                    if results:
                        print(f"\nTop result:")
                        print(f"      {results[0].get('title', 'N/A')}")
                        print(f"      {results[0].get('snippet', 'N/A')[:100]}...")
            else:
                print(f"Status: {context.execution.action_status.value}")
                if context.execution.error:
                    print(f"Error: {context.execution.error}")
            print()
        
        # === STEP 5: MEMORY STORAGE ===
        if verbose:
            print("STEP 5: MEMORY STORAGE")
            print("─"*80)
        
        self.memory_manager.store_experience(context)
        
        if verbose:
            total_exp = len(self.memory_manager.long_term.experiences)
            print(f"Experience stored")
            print(f"Total experiences: {total_exp}")
            print()
        
        # === DISPLAY COMPLETE CONTEXT ===
        if verbose:
            print(context.pretty_print())
        
        # Return structured result
        return {
            "question": question,
            "answer": self._format_answer(context),
            "success": context.execution.action_status.value == "completed",
            "confidence": context.reasoning.confidence if context.reasoning else 0.0,
            "context": context
        }
    
    def _format_answer(self, context: AgentContext) -> str:
        """Format answer from execution result"""
        if not context.execution or not context.execution.result:
            return "I couldn't find an answer to your question."
        
        result = context.execution.result
        
        # Handle Google Search results
        if isinstance(result, dict) and "results" in result:
            results = result["results"]
            if not results:
                return "I couldn't find any relevant results for your question."
            
            # Format top 3 results
            answer_parts = []
            answer_parts.append(f"I found {len(results)} results:\n")
            
            for i, res in enumerate(results[:3], 1):
                answer_parts.append(f"{i}. {res.get('title', 'N/A')}")
                answer_parts.append(f"{res.get('snippet', 'N/A')}")
                answer_parts.append(f"{res.get('url', 'N/A')}\n")
            
            return "\n".join(answer_parts)
        
        return str(result)
    
    def chat(self):
        """
        Interactive chat mode
        Allows continuous conversation with the agent
        """
        print("\n" + "="*80)
        print("Q&A Agent - Interactive Mode")
        print("="*80)
        print("\nType your questions below. Commands:")
        print("  'quit' or 'exit' - Exit chat")
        print("  'memory' - Show memory statistics")
        print("  'clear' - Clear short-term memory\n")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'memory':
                    self._show_memory_stats()
                    continue
                
                if question.lower() == 'clear':
                    self.memory_manager.clear_short_term()
                    print("✓ Short-term memory cleared\n")
                    continue
                
                # Process question
                result = self.ask(question, verbose=False)
                
                # Display answer
                print(f"\nAgent: {result['answer']}\n")
                print(f"   (Confidence: {result['confidence']:.1%})\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def _show_memory_stats(self):
        """Display memory statistics"""
        summary = self.memory_manager.get_memory_summary()
        
        print("\n" + "="*80)
        print("MEMORY STATISTICS")
        print("="*80)
        
        # Short-term
        st = summary['short_term']
        print(f"\nShort-Term Memory:")
        print(f"   Size: {st['size']}/{st['max_size']}")
        
        # Long-term
        lt = summary['long_term']
        print(f"\nLong-Term Memory:")
        print(f"   Total experiences: {lt['total_experiences']}")
        print(f"   Successful: {lt['successful']}")
        print(f"   Failed: {lt['failed']}")
        
        # Patterns
        if 'patterns' in summary and summary['patterns']:
            patterns = summary['patterns']
            if 'success_patterns' in patterns:
                sp = patterns['success_patterns']
                if 'success_rate' in sp:
                    print(f"\nSuccess Rate: {sp['success_rate']:.1%}")
        
        print("="*80 + "\n")


def main():
    """
    Main entry point
    Demonstrates different usage modes
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Q&A Agent with Google Search")
    parser.add_argument(
        "--mode",
        choices=["single", "chat", "demo"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (for single mode)"
    )
    parser.add_argument(
        "--no-google",
        action="store_true",
        help="Use simulated search instead of real Google API"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = QAAgent(use_real_google_search=not args.no_google)
    
    if args.mode == "single":
        # Single question mode
        question = args.question or "What is Python programming language?"
        result = agent.ask(question)
        
    elif args.mode == "chat":
        # Interactive chat mode
        agent.chat()
        
    else:  # demo mode
        # Demo mode: Ask multiple questions
        demo_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the benefits of Python programming?"
        ]
        
        print("\n" + "="*80)
        print("DEMO MODE: Processing multiple questions")
        print("="*80 + "\n")
        
        for i, q in enumerate(demo_questions, 1):
            print(f"\n{'='*80}")
            print(f"Question {i}/{len(demo_questions)}")
            print(f"{'='*80}\n")
            
            result = agent.ask(q, verbose=True)
            
            input("\n⏸Press Enter to continue to next question...")
        
        print("\n" + "="*80)
        print("Demo completed!")
        print("="*80)
        
        # Show final memory stats
        agent._show_memory_stats()


if __name__ == "__main__":
    main()
