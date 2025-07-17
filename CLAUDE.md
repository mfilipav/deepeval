# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepEval is an open-source LLM evaluation framework built with Python and Poetry. It provides a comprehensive suite of metrics and tools for evaluating large language models, similar to Pytest but specialized for LLM outputs. The framework supports both end-to-end and component-level evaluation.

## Common Development Commands

### Installation and Setup
```bash
# Install dependencies
poetry install

# For development with root project
poetry install --no-interaction
```

### Testing
```bash
# Run all tests (standard pytest)
poetry run pytest tests/ --ignore=tests/test_deployment.py --ignore=tests/test_benchmarks.py --ignore=tests/test_hybrid_tracing.py --ignore=tests/test_synthesizer.py

# Run a single test file with pytest
pytest tests/test_<specific_test>.py

# Run DeepEval-specific tests (preferred for LLM evaluation tests)
deepeval test run test_example.py

# Run tests in parallel
deepeval test run test_file.py -n 4

# Run with verbose output
deepeval test run test_file.py --verbose

# Run with caching
deepeval test run test_file.py --use-cache
```

### Code Formatting
```bash
# Format code (line length: 80 characters)
black .

# Check formatting without changes
black --check --verbose .
```

### DeepEval CLI
```bash
# Login to DeepEval platform
deepeval login

# Set up different model providers
deepeval set-azure-openai --openai-api-key <key> --openai-endpoint <endpoint>
deepeval set-ollama <model_name> --base-url http://localhost:11434
deepeval set-gemini --model-name <name> --google-api-key <key>

# View test results
deepeval view
```

## Architecture Overview

### Core Components

**Metrics System** (`deepeval/metrics/`):
- **G-Eval**: Research-backed evaluation using LLMs with customizable criteria
- **RAG Metrics**: Answer relevancy, faithfulness, contextual precision/recall/relevancy
- **Agentic Metrics**: Task completion, tool correctness
- **Conversational Metrics**: Knowledge retention, conversation completeness
- **Safety Metrics**: Hallucination, bias, toxicity detection
- **Multimodal Metrics**: Image-text evaluation capabilities

**Models Integration** (`deepeval/models/`):
- LLM providers: OpenAI, Anthropic, Gemini, Azure, Ollama, LiteLLM, Amazon Bedrock
- Embedding models: OpenAI, Azure, Ollama, local models
- Specialized models: Detoxify, SummaC for specific evaluation tasks

**Test Framework** (`deepeval/test_case/`):
- `LLMTestCase`: Standard single-turn evaluation
- `ConversationalTestCase`: Multi-turn conversation evaluation
- `ArenaTestCase`: Head-to-head model comparison
- `MLLMTestCase`: Multimodal (text + image) evaluation

**Evaluation Engine** (`deepeval/evaluate/`):
- End-to-end evaluation pipeline
- Component-level evaluation with tracing
- Batch evaluation for datasets
- Integration with pytest framework

**Synthesizer** (`deepeval/synthesizer/`):
- Synthetic dataset generation from documents
- Context-based test case creation
- Golden dataset generation for benchmarking

### Key Architectural Patterns

**Plugin System**: Metrics are modular and can be combined. Each metric implements a standard interface with `measure()` method.

**Tracing System** (`deepeval/tracing/`): Non-intrusive component-level evaluation using `@observe` decorator to trace LLM calls, retrievers, and agents.

**Provider Abstraction**: Unified interface for different LLM providers through base model classes, allowing easy switching between providers.

**Async Support**: Most evaluation operations support both synchronous and asynchronous execution for performance.

## Testing Patterns

### Test File Structure
- Test files must start with `test_` prefix
- Use `LLMTestCase` for single evaluations
- Use `EvaluationDataset` for bulk evaluations
- Metrics are applied using `assert_test()` or `evaluate()`

### Example Test Pattern
```python
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def test_case():
    metric = GEval(
        name="Correctness",
        criteria="Determine if the output is correct",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="...",
        actual_output="...",
        expected_output="..."
    )
    assert_test(test_case, [metric])
```

## Development Notes

- **Poetry**: Primary dependency manager, use `poetry run` for commands
- **Python Version**: Supports Python >=3.9, <4.0
- **Code Style**: Black formatting with 80-character line length
- **Testing**: Built on pytest with DeepEval extensions
- **Environment Variables**: Most LLM providers require API keys set as environment variables
- **Platform Integration**: Optional integration with Confident AI platform for test result management and sharing