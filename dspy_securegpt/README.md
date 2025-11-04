# SecureGPT + DSPy Integration

This integration allows you to use SecureGPT models with DSPy through a custom LiteLLM handler.

## Overview

DSPy is a framework for programming language models. This integration bridges SecureGPT's chat API with DSPy's LM interface, enabling you to:

- Use SecureGPT models with all DSPy features
- Leverage DSPy's Chain-of-Thought, ReAct, and other reasoning patterns
- Benefit from DSPy's optimization and compilation capabilities
- Use SecureGPT's enterprise features (authentication, rate limiting, etc.)

## Architecture

```
┌─────────────┐
│    DSPy     │
│   Modules   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  dspy.LM    │
│  Interface  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LiteLLM    │
│  (routing)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ SecureGPTLiteLLM    │
│ Handler (custom)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  SecureGPTChat      │
│  (LangChain-based)  │
└─────────────────────┘
```

## Installation

The integration is included in this workspace. Make sure you have the required dependencies:

```bash
# Using uv (recommended for this workspace)
uv sync

# Or using pip
pip install dspy cai-securegpt-client
```

## Quick Start

### Basic Usage

```python
import dspy
from dspy_securegpt.config import configure_securegpt_lm

# Import to register the custom handler
import dspy_securegpt.securegpt_litellm

# Configure SecureGPT LM
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    temperature=0.7
)

# Set as default LM
dspy.configure(lm=lm)

# Use it!
response = lm("What is machine learning?")
print(response)
```

### With DSPy Signatures

```python
import dspy
from dspy_securegpt.config import configure_securegpt_lm
import dspy_securegpt.securegpt_litellm

# Configure
lm = configure_securegpt_lm(model_id="gpt-4-turbo-2024-04-09")
dspy.configure(lm=lm)

# Define a signature
class QA(dspy.Signature):
    """Answer questions with reasoning."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Use Chain of Thought
cot = dspy.ChainOfThought(QA)
response = cot(question="What are the benefits of neural networks?")
print(response.answer)
```

## Configuration

### Environment Variables

You can configure SecureGPT using environment variables:

```bash
export SECUREGPT_URL="https://api.se.axa-go.applications.services.axa-tech.intraxa/ago-m365-securegpt-hub-v1-vrs"
export SECUREGPT_MODEL_ID="gpt-4-turbo-2024-04-09"
export SECUREGPT_PROVIDER="openai"
export SECUREGPT_TEMPERATURE="0.7"
export SECUREGPT_MAX_TOKENS="4000"
```

Then use the default configuration:

```python
from dspy_securegpt.config import get_default_securegpt_lm
import dspy_securegpt.securegpt_litellm

lm = get_default_securegpt_lm()
dspy.configure(lm=lm)
```

### Programmatic Configuration

```python
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",  # Model to use
    provider="openai",                   # Provider (openai, mistral, etc.)
    temperature=0.7,                     # Sampling temperature
    max_tokens=4000,                     # Max tokens to generate
    api_base="https://...",              # API base URL (optional)
    api_key="your-token",                # Access token (optional)
)
```

## Authentication

The integration uses SecureGPT's authentication:

1. **OneLogin (default)**: Automatically authenticates using OneLogin
2. **Access Token**: Pass `api_key` parameter if you have a token

```python
# Using OneLogin (default)
lm = configure_securegpt_lm(model_id="gpt-4-turbo-2024-04-09")

# Using access token
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    api_key="your-access-token"
)
```

## Available Models

You can use any model available in SecureGPT:

- `gpt-4-turbo-2024-04-09` (default)
- `gpt-4o`
- `gpt-3.5-turbo`
- Mistral models
- Other models supported by your SecureGPT instance

## Features

### Supported DSPy Features

✅ **Basic Completion**: Simple text generation  
✅ **Signatures**: Structured input/output with DSPy signatures  
✅ **Chain of Thought**: Multi-step reasoning  
✅ **Predict**: Structured predictions  
✅ **ReAct**: Reasoning and acting patterns  
✅ **Temperature Control**: Adjust randomness  
✅ **Token Limits**: Control output length  

### Supported SecureGPT Features

✅ **Authentication**: OneLogin and token-based auth  
✅ **Rate Limiting**: Automatic retry with exponential backoff  
✅ **Error Handling**: Robust error handling and retries  
✅ **Multiple Providers**: OpenAI, Mistral, and other providers  
✅ **Streaming**: Streaming responses (experimental)  
✅ **Tool Calling**: Function/tool calling support  

## Examples

See `example_securegpt_dspy.py` for comprehensive examples:

```bash
python dspy_securegpt/example_securegpt_dspy.py
```

Examples include:
- Basic Q&A
- Chain of Thought reasoning
- Multi-hop reasoning
- Structured outputs
- Temperature comparisons

## Advanced Usage

### Custom DSPy Module

```python
import dspy

class ComplexQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Use with SecureGPT
lm = configure_securegpt_lm(model_id="gpt-4-turbo-2024-04-09")
dspy.configure(lm=lm)

qa = ComplexQA()
answer = qa(question="What is quantum computing?")
```

### Multiple Models

```python
# Use different models for different tasks
fast_lm = configure_securegpt_lm(
    model_id="gpt-3.5-turbo",
    temperature=0.3
)

smart_lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    temperature=0.7
)

# Use fast model for simple tasks
with dspy.context(lm=fast_lm):
    quick_answer = lm("What is 2+2?")

# Use smart model for complex tasks
with dspy.context(lm=smart_lm):
    detailed_answer = lm("Explain quantum entanglement")
```

## Troubleshooting

### Import Error

If you get an import error, make sure the handler is registered:

```python
import dspy_securegpt.securegpt_litellm  # This registers the handler
```

### Authentication Error

Ensure you have:
- OneLogin credentials configured, OR
- A valid access token passed via `api_key` parameter

### Rate Limiting

SecureGPT includes automatic retry with exponential backoff. If you hit rate limits frequently:

```python
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    max_retries=5  # Increase retry attempts
)
```

### Connection Issues

Check your network and API base URL:

```python
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    api_base="https://your-securegpt-instance.com",
    timeout=120  # Increase timeout
)
```

## Implementation Details

### How It Works

1. **Registration**: `securegpt_litellm.py` registers a custom LiteLLM handler
2. **Model String**: Uses format `securegpt/model-id` for routing
3. **Message Conversion**: Converts between LiteLLM and LangChain message formats
4. **Response Mapping**: Transforms SecureGPT responses to OpenAI-compatible format
5. **Client Caching**: Reuses SecureGPTChat clients for efficiency

### Files

- `dspy_securegpt/securegpt_litellm.py`: Custom LiteLLM handler
- `dspy_securegpt/config.py`: Configuration utilities
- `dspy_securegpt/example_securegpt_dspy.py`: Usage examples
- `dspy_securegpt/README.md`: This file

## Contributing

To improve the integration:

1. Test with your SecureGPT instance
2. Report issues or edge cases
3. Suggest enhancements
4. Add more examples

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [SecureGPT Client](../packages/cai-securegpt-client/)
- [LangChain Documentation](https://python.langchain.com/)

## License

Same as the parent project.

