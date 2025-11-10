#!/usr/bin/env python3
"""
Example: Using Agent Orchestrator
==================================

This example demonstrates the difference between:
1. Manual agent loop (calling each component)
2. Orchestrator (automated agent loop)

The orchestrator simplifies agent usage and ensures correct execution order.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_context import AgentContext
from perception.text.text_perception import TextPerceptionSystem
from memory.memory_manager import MemoryManager
from reasoning.llm_reasoner import LLMReasoningSystem
from execution.action_executor import ActionExecutor
from execution.tools.tool_registry import ToolRegistry
from execution.tools.builtin_tools import SearchTool
from orchestration.agent_orchestrator import AgentOrchestrator
from utils.llm_client import LLMClient


def manual_agent_loop(question: str):
    """
    Manual way: Call each component step by step
    
    Pros: Full control, explicit
    Cons: Verbose, error-prone, easy to forget steps
    """
    print("\n" + "="*80)
    print("MANUAL AGENT LOOP")
    print("="*80)
    print("Calling each component manually...\n")
    
    # Setup
    perception = TextPerceptionSystem()
    memory = MemoryManager()
    llm = LLMClient()
    reasoning = LLMReasoningSystem(llm)
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    execution = ActionExecutor(tool_registry)
    
    # Manual loop
    context = AgentContext()
    
    print("Step 1: Calling perception.perceive()...")
    context = perception.perceive(question, context)
    
    print("Step 2: Calling memory.retrieve_context()...")
    context = memory.retrieve_context(question, context)
    
    print("Step 3: Calling reasoning.reason()...")
    context = reasoning.reason(context)
    
    print("Step 4: Calling execution.execute()...")
    context = execution.execute(context)
    
    print("Step 5: Calling memory.store_experience()...")
    memory.store_experience(context)
    
    print("\nManual loop complete!")
    print(f"   Result: {context.execution.result}")
    
    return context


def orchestrated_agent_loop(question: str):
    """
    Orchestrator way: One line to run complete loop
    
    Pros: Simple, clean, handles errors, consistent
    Cons: Less control over individual steps
    """
    print("\n" + "="*80)
    print("ORCHESTRATED AGENT LOOP")
    print("="*80)
    print("Using orchestrator for automated execution...\n")
    
    # Setup
    perception = TextPerceptionSystem()
    memory = MemoryManager()
    llm = LLMClient()
    reasoning = LLMReasoningSystem(llm)
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    execution = ActionExecutor(tool_registry)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        perception=perception,
        reasoning=reasoning,
        memory=memory,
        execution=execution,
        verbose=True
    )
    
    # Single line execution!
    result = orchestrator.run(question, max_iterations=1)
    
    print(f"\n Orchestrated loop complete!")
    print(f"   Success: {result['success']}")
    print(f"   Iterations: {result['total_iterations']}")
    print(f"   Result: {result['final_result']}")
    
    return result['context']


def multi_iteration_example():
    """
    Demonstrate multi-iteration with orchestrator
    
    Useful when agent needs multiple steps to complete a task
    """
    print("\n" + "="*80)
    print("MULTI-ITERATION EXAMPLE")
    print("="*80)
    
    # Setup
    perception = TextPerceptionSystem()
    memory = MemoryManager()
    llm = LLMClient()
    reasoning = LLMReasoningSystem(llm)
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    execution = ActionExecutor(tool_registry)
    
    orchestrator = AgentOrchestrator(
        perception=perception,
        reasoning=reasoning,
        memory=memory,
        execution=execution,
        verbose=True
    )
    
    # Run with multiple iterations
    result = orchestrator.run(
        "Search for Python tutorials and summarize the top 3",
        max_iterations=3,
        stop_on_completion=True
    )
    
    print("\n" + "="*80)
    print(f"Multi-iteration complete!")
    print(f"  Total iterations: {result['total_iterations']}")
    print(f"  Success: {result['success']}")
    print("="*80)


def comparison_demo():
    """
    Side-by-side comparison of both approaches
    """
    question = "What is artificial intelligence?"
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "ORCHESTRATOR vs MANUAL COMPARISON" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    # Manual approach
    print("\n\n" + "─"*80)
    print("APPROACH 1: Manual (Traditional)")
    print("─"*80)
    print("""
Code:
    context = AgentContext()
    context = perception.perceive(question, context)
    context = memory.retrieve_context(question, context)
    context = reasoning.reason(context)
    context = execution.execute(context)
    memory.store_experience(context)
    
Pros: Full control, explicit steps
Cons: Verbose, error-prone, repetitive
Lines: 6+
""")
    
    try:
        context1 = manual_agent_loop(question)
        manual_success = True
    except Exception as e:
        print(f"❌ Error: {e}")
        manual_success = False
    
    # Orchestrator approach
    print("\n\n" + "─"*80)
    print("APPROACH 2: Orchestrator (Modern)")
    print("─"*80)
    print("""
Code:
    orchestrator = AgentOrchestrator(perception, reasoning, memory, execution)
    result = orchestrator.run(question)
    
Pros: Clean, simple, error handling built-in
Cons: Less direct control
Lines: 2
""")
    
    try:
        context2 = orchestrated_agent_loop(question)
        orchestrator_success = True
    except Exception as e:
        print(f"❌ Error: {e}")
        orchestrator_success = False
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"""
┌─────────────────────┬──────────────┬────────────────┐
│ Metric              │ Manual       │ Orchestrator   │
├─────────────────────┼──────────────┼────────────────┤
│ Lines of code       │ 6+           │ 2              │
│ Error handling      │ Manual       │ Built-in       │
│ Logging             │ Manual       │ Automatic      │
│ Easy to use         │ No           │ Yes            │
│ Maintainable        │ Medium       │ High           │
│ Success             │ {('True' if manual_success else 'False')}           │ {('✅' if orchestrator_success else '❌')}              │
└─────────────────────┴──────────────┴────────────────┘

🎯 Recommendation: Use Orchestrator for production code!
""")


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Orchestrator Examples")
    parser.add_argument(
        "--mode",
        choices=["comparison", "manual", "orchestrator", "multi"],
        default="comparison",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "comparison":
            comparison_demo()
        elif args.mode == "manual":
            manual_agent_loop("What is machine learning?")
        elif args.mode == "orchestrator":
            orchestrated_agent_loop("What is deep learning?")
        elif args.mode == "multi":
            multi_iteration_example()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
