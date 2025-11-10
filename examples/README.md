# Examples

This directory contains practical examples demonstrating how to use the Autonomous AI Agent system.

## Available Examples

### 1. Q&A Agent with Google Search (`qa_agent_example.py`)

A complete autonomous agent that can answer questions using Google Search.

**Features:**
- ✅ Real Google Search integration
- ✅ Text perception with NLP
- ✅ Memory system (learns from past Q&As)
- ✅ LLM-based reasoning
- ✅ Multiple operation modes

**Usage:**

```bash
# Demo mode (default) - Ask 3 questions with full details
python examples/qa_agent_example.py --mode demo

# Interactive chat mode
python examples/qa_agent_example.py --mode chat

# Single question mode
python examples/qa_agent_example.py --mode single --question "What is AI?"

# Use simulated search (no Google API needed)
python examples/qa_agent_example.py --no-google
```

**Architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                     Q&A Agent                            │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Perception│   │  Memory  │   │Reasoning │
    │  (Text)  │   │(ST + LT) │   │  (LLM)   │
    └──────────┘   └──────────┘   └──────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │   Execution    │
                  │ (Google Search)│
                  └────────────────┘
```

**Code Example:**

```python
from examples.qa_agent_example import QAAgent

# Initialize agent
agent = QAAgent(use_real_google_search=True)

# Ask a question
result = agent.ask("What is Python?", verbose=True)

# Get answer
print(result['answer'])
print(f"Confidence: {result['confidence']:.1%}")

# Interactive mode
agent.chat()
```

## Output Examples

### Demo Mode Output

```
================================================================================
🤖 Initializing Q&A Agent...
================================================================================

📥 Setting up Perception...
   ✓ Text perception ready

💾 Setting up Memory...
   ✓ Memory system ready (short-term: 10, long-term: unlimited)

🧠 Setting up Reasoning...
✓ LLM Client initialized
  Model: gemini-2.0-flash-lite
  Endpoint: https://api2.key4u.shop/v1
   ✓ LLM reasoning ready

⚙️  Setting up Execution...
✓ Google Search Tool initialized (CSE ID: a02b9c97...)
   ✓ Registered: google_search (Real Google API)
   ✓ Execution system ready

================================================================================
✅ Q&A Agent initialized successfully!
================================================================================

🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵
❓ Question: What is artificial intelligence?
🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵

📥 STEP 1: PERCEPTION
────────────────────────────────────────────────────────────────────────────────
   Intent detected: query
   Keywords: artificial, intelligence
   Confidence: 95.0%

💾 STEP 2: MEMORY RETRIEVAL
────────────────────────────────────────────────────────────────────────────────
   Retrieved 0 similar past experiences

🧠 STEP 3: REASONING
────────────────────────────────────────────────────────────────────────────────
   Decided action: google_search
   Parameters: {'query': 'What is artificial intelligence?'}
   Confidence: 92.5%

⚙️  STEP 4: EXECUTION
────────────────────────────────────────────────────────────────────────────────
   Status: ✅ Completed
   Found 5 results

   📝 Top result:
      Artificial Intelligence (AI) | IBM
      Artificial intelligence (AI) is technology that enables computers and machines...

💾 STEP 5: MEMORY STORAGE
────────────────────────────────────────────────────────────────────────────────
   Experience stored
   Total experiences: 1
```

### Pretty Print Output

```
════════════════════════════════════════════════════════════════════════════════
║  🤖 AGENT CONTEXT - Iteration 1                                              ║
════════════════════════════════════════════════════════════════════════════════

📥 PERCEPTION
────────────────────────────────────────────────────────────────────────────────
  Modality:      TEXT
  Raw Input:     What is artificial intelligence?...
  Intent:        query
  Keywords:      artificial, intelligence
  Complexity:    simple
  Objects:       0 detected
  Confidence:    95.0%
  Process Time:  0.52ms

💾 MEMORY
────────────────────────────────────────────────────────────────────────────────
  Similar Exp:   0 retrieved
  Recent Acts:   0 in buffer
  Retrieval:     0.15ms

🧠 REASONING
────────────────────────────────────────────────────────────────────────────────
  Method:        chain_of_thought
  Goal:          Not set
  Action Plan:   1 step(s)
                 1. google_search(query=What is artificial intelligen...)
  Thinking:      The user is asking about artificial intelligence. I ...
  Confidence:    🟢 92.5%
  Process Time:  1245.32ms

⚙️  EXECUTION
────────────────────────────────────────────────────────────────────────────────
  Status:        ✅ COMPLETED
  Tool:          google_search
  Result:        {'query': 'What is artificial intelligence?', 'resu...
  Exec Time:     342.15ms

════════════════════════════════════════════════════════════════════════════════
```

## Building Your Own Agent

Follow this template to create custom agents:

```python
from core.agent_context import AgentContext
from perception.text.text_perception import TextPerceptionSystem
from memory.memory_manager import MemoryManager
from reasoning.llm_reasoner import LLMReasoningSystem
from execution.action_executor import ActionExecutor
from execution.tools.tool_registry import ToolRegistry
from utils.llm_client import LLMClient

class MyCustomAgent:
    def __init__(self):
        # 1. Perception
        self.perception = TextPerceptionSystem()
        
        # 2. Memory
        self.memory = MemoryManager()
        
        # 3. Reasoning
        llm = LLMClient()
        self.reasoning = LLMReasoningSystem(llm)
        
        # 4. Execution with custom tools
        self.registry = ToolRegistry()
        # Register your custom tools here
        self.executor = ActionExecutor(self.registry)
    
    def process(self, input_data):
        context = AgentContext()
        
        # Run the autonomous loop
        context = self.perception.perceive(input_data, context)
        context = self.memory.retrieve_context(input_data, context)
        context = self.reasoning.reason(context)
        context = self.executor.execute(context)
        self.memory.store_experience(context)
        
        return context
```

## Key Concepts Demonstrated

### 1. **Modular Architecture**
- Each component (Perception, Memory, Reasoning, Execution) is independent
- Easy to swap implementations
- Clear separation of concerns

### 2. **Context Passing**
- `AgentContext` is the communication channel
- Each system enriches the context
- Full audit trail available

### 3. **Memory & Learning**
- Experiences stored automatically
- Pattern learning from success/failure
- Retrieval-Augmented Generation (RAG)

### 4. **Tool Integration**
- Tools extend agent capabilities
- Easy to add custom tools
- Type-safe parameter validation

### 5. **Observability**
- Detailed logging at each step
- Pretty-printed context visualization
- Performance metrics

## Requirements

```bash
# Core requirements
pip install openai python-dotenv requests

# For Google Search
# Set in .env:
GOOGLE_SEARCH_ENGINE_KEY=your_key
GOOGLE_ID_CSE=your_cse_id
```

## Next Steps

1. **Try the examples**: Run `qa_agent_example.py` in different modes
2. **Study the code**: See how components work together
3. **Build custom tools**: Extend functionality
4. **Create your agent**: Use the template above
5. **Monitor & debug**: Use `pretty_print()` for visibility

## Troubleshooting

**Q: "Google Search credentials not found"**
- Check `.env` file has `GOOGLE_SEARCH_ENGINE_KEY` and `GOOGLE_ID_CSE`
- Run with `--no-google` flag to use simulated search

**Q: "Could not initialize LLM"**
- Check `.env` file has `API_KEY`, `BASE_URL`, `MODEL_NAME`
- Agent will fallback to simple reasoning

**Q: "Tool not found"**
- Ensure tool is registered: `registry.register(your_tool)`
- Check tool name matches what reasoning system generates

## Contributing

To add more examples:
1. Create new file in `examples/`
2. Follow the pattern in `qa_agent_example.py`
3. Update this README
4. Add test in `tests/`

## License

MIT License - See LICENSE file for details
