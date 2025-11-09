# reasoning/llm_reasoner.py
"""
LLM-Based Reasoning Implementation
Uses OpenAI/Anthropic API for reasoning
"""

from typing import Any, Dict, Optional
import json
import re
from core.agent_context import AgentContext, ReasoningData, ReasoningMethod
from reasoning.base import ReasoningSystem, ActionPlan, PromptBuilder

class LLMReasoningSystem(ReasoningSystem):
    """
    Paper-Aligned LLM Reasoning
    - Chain-of-Thought for step-by-step planning
    - Tree-of-Thought for multi-path exploration  
    - Adaptive reasoning from feedback
    """
    
    def __init__(self, llm_client, tool_registry=None):
        """
        Args:
            llm_client: OpenAI or Anthropic client
        """
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.reasoning_history = []

    def _build_reasoning_prompt(self, context: AgentContext) -> str:
        """Build prompt with tool information"""
        
        perception_str = json.dumps(context.perception.to_dict(), indent=2)
        
        tools_section = ""
        if self.tool_registry:
            tools_section = f"""
## AVAILABLE TOOLS
You can use these tools by specifying tool name and parameters in the 'action' part of your JSON response.

{self.tool_registry.get_all_descriptions()}
"""
        
        prompt = f"""
You are an autonomous agent analyzing a situation. Your goal is to decide the next best action.

## CURRENT PERCEPTION
{perception_str}

## MEMORY CONTEXT
(Similar past experiences and learned patterns here)

{tools_section}

## TASK
Based on the current situation, what is your next action?
Think step-by-step and provide your action plan in a JSON format.

## OUTPUT FORMAT
You MUST respond with a valid JSON object in this exact format (no markdown, no extra text):
{{
  "thinking": "Your step-by-step reasoning here. Explain your thought process for choosing the action.",
  "action": {{
    "tool": "tool_name",
    "params": {{"param1": "value1"}}
  }},
  "confidence": 0.85
}}

IMPORTANT: 
- Start your response directly with opening brace and end with closing brace
- Do not include markdown code blocks (```json) or any other formatting
- The confidence value must be between 0.0 and 1.0
- If unsure what tool to use, use "respond" tool with a message parameter
"""
        return prompt
    
    def reason(self, context: AgentContext) -> AgentContext:
        """
        Main reasoning loop
        """
        context.log_step("Reasoning", "Starting reasoning phase")
        
        prompt = self._build_reasoning_prompt(context)
        context.log_step("Reasoning", f"Built prompt ({len(prompt)} chars)")
        
        reasoning_response = self.chain_of_thought(prompt)
        context.log_step("Reasoning", "LLM generated reasoning")
        
        action_plan = self._parse_action_plan(reasoning_response)
        
        context.reasoning = ReasoningData(
            action_plan=action_plan.steps,
            reasoning_trace=action_plan.reasoning_trace,
            confidence=action_plan.confidence,
            reasoning_method=ReasoningMethod.CHAIN_OF_THOUGHT
        )
        
        self.reasoning_history.append(action_plan)
        
        return context
    
    def chain_of_thought(self, prompt: str) -> Dict:
        """
        Executes a chain-of-thought prompt and returns the parsed JSON response.
        """
        messages = [
            {
                "role": "system",
                "content": "You are an autonomous agent. Think step-by-step and respond ONLY with valid JSON. Do not include any text before or after the JSON object."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self.llm.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Try to extract JSON from response (handle cases where LLM adds extra text)
        try:
            # First try direct parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response text
            try:
                # Look for JSON object between curly braces
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # Handle cases where the LLM fails to produce valid JSON
            return {
                "thinking": f"LLM response was not valid JSON. Raw response: {response_text[:200]}",
                "action": {"tool": "error", "params": {"message": "Invalid LLM response format."}},
                "confidence": 0.0
            }

    def tree_of_thought(self, prompt: str, num_paths: int = 3) -> Dict:
        # This method would also need to be updated to handle JSON parsing
        # For now, we focus on the primary chain_of_thought method
        pass
    
    def adapt_to_feedback(self, 
                         previous_action: Dict,
                         feedback: str,
                         context: AgentContext) -> ActionPlan:
        pass

    def _parse_action_plan(self, reasoning_response: Dict) -> ActionPlan:
        """Parse the JSON response from the LLM into a structured ActionPlan."""
        thinking = reasoning_response.get("thinking", "")
        action = reasoning_response.get("action", {})
        confidence = reasoning_response.get("confidence", 0.0)
        
        steps = [action] if action else []
        
        return ActionPlan(
            steps=steps,
            reasoning_trace=thinking.split('\n'),
            confidence=confidence
        )

    def _compute_confidence(self, thinking: str) -> float:
        """Estimate confidence from thinking text"""
        confidence_markers = ["certain", "confident", "clear", "definitely"]
        uncertainty_markers = ["uncertain", "might", "perhaps", "maybe"]
        
        text_lower = thinking.lower()
        
        confidence = 0.5
        for marker in confidence_markers:
            if marker in text_lower:
                confidence += 0.1
        
        for marker in uncertainty_markers:
            if marker in text_lower:
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


class LLMPromptBuilder(PromptBuilder):
    """Build prompts for LLM reasoning"""
    
    @staticmethod
    def build_reasoning_prompt(context: AgentContext) -> str:
        """Build complete reasoning prompt with all context"""
        
        perception_str = json.dumps(context.perception.to_dict(), indent=2)
        
        memory_context = context.memory
        similar_exp = memory_context.similar_experiences
        patterns = memory_context.learned_patterns
        
        prompt = f"""
You are an autonomous agent analyzing a situation.

## CURRENT PERCEPTION
{perception_str}

## SIMILAR PAST EXPERIENCES
{json.dumps(similar_exp, indent=2) if similar_exp else "No relevant past experiences"}

## LEARNED PATTERNS  
{json.dumps(patterns, indent=2) if patterns else "No patterns yet"}

## TASK
Based on the current perception and past experiences, what should be your next action?

Think step-by-step:
1. What is the current situation?
2. What did work in similar situations before?
3. What didn't work?
4. What is your plan?

Generate your action plan in the specified JSON format.
"""
        return prompt
    
    @staticmethod
    def build_cot_prompt(perception: Dict, memory: Dict) -> str:
        pass
    
    @staticmethod
    def build_tot_prompt(perception: Dict, memory: Dict) -> str:
        pass
