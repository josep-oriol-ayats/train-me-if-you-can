## Custom Langchain LLM and Embeddings Library for SecureGPT

Welcome to SecureGPT Langchain, a custom LLM and Embeddings implementation for SecureGPT compatible with LangChain python API.

Based on code from https://github.axa.com/axa-group-genai/SmartGuide, thanks for your work :)


### Features

 * Custom implementations of the LangChain _BaseChatModel_ and _Embeddings_ for SecureGPT.
 * Easy integration with _SecureGPT_ services.
 * _OneLogin_ authentication management.
 * Flexible and extensible codebase for future enhancements.


### Installation

To install the library, you can use _poetry_. Make sure you have Python 3.11 or higher installed.

```bash
poetry install
```

### Usage

Here’s a quick example of how to use the library:

#### Basic Chat Examples

##### Simple Single Message
```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

# Initialize the chat model
chat_model = SecureGPTChat(
    temperature=0.7,
    max_retries=2,
    timeout=60
)

# Simple message
message = HumanMessage(content="Tell me a short joke about programming.")
response = chat_model.invoke([message])
print(f"Response: {response.content}")
```

##### Multi-turn Conversation
```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_model = SecureGPTChat(temperature=0.5)

# Start with a system message to set context
messages = [
    SystemMessage(content="You are a helpful assistant that gives concise, friendly responses."),
    HumanMessage(content="Hi! Can you explain what Python is?")
]

# First exchange
response1 = chat_model.invoke(messages)
print(f"Assistant: {response1.content}")

# Continue the conversation
messages.append(AIMessage(content=response1.content))
messages.append(HumanMessage(content="What are some popular Python frameworks?"))

response2 = chat_model.invoke(messages)
print(f"Assistant: {response2.content}")
```

##### Streaming Responses
```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage
message = HumanMessage(content="Write a short story about a robot learning to cook.")
chat_model = SecureGPTChat()

print("Streaming response:")
for chunk in chat_model.stream([message]):
    if chunk.content:
        print(chunk.content, end='', flush=True)
```

##### Async Operations
```python
import asyncio
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat

chat_model = SecureGPTChat()

async def async_chat():
    message = HumanMessage(content="What are the benefits of asynchronous programming?")
    response = await chat_model.ainvoke([message])
    print(f"Async response: {response.content}")

# Run async example
asyncio.run(async_chat())
```

##### Different Temperature Settings
```python
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat

# More deterministic responses (temperature 0.1-0.3)
deterministic_chat = SecureGPTChat(temperature=0.1)

# More creative responses (temperature 0.7-0.9)
creative_chat = SecureGPTChat(temperature=0.9)

prompt = "Complete this sentence: The future of artificial intelligence is"
message = HumanMessage(content=prompt)

det_response = deterministic_chat.invoke([message])
creative_response = creative_chat.invoke([message])

print(f"Deterministic: {det_response.content}")
print(f"Creative: {creative_response.content}")
```

#### Structured Output Examples

##### Basic Structured Output with Pydantic
```python
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat

class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

# Create structured output model
chat_model = SecureGPTChat()
structured_llm = chat_model.with_structured_output(Joke)

# Generate structured joke
messages = [HumanMessage(content="Tell me a joke about cats")]
joke = structured_llm.invoke(messages)

print(f"Setup: {joke.setup}")
print(f"Punchline: {joke.punchline}")
print(f"Rating: {joke.rating}/10")
```

##### Complex Structured Output
```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat

class Recipe(BaseModel):
    """A cooking recipe."""
    name: str = Field(description="Name of the dish")
    ingredients: list[str] = Field(description="List of ingredients needed")
    instructions: list[str] = Field(description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    
    
chat_model = SecureGPTChat()

structured_llm = chat_model.with_structured_output(Recipe)
messages = [HumanMessage(content="Give me a simple recipe for chocolate chip cookies")]

recipe = structured_llm.invoke(messages)
print(f"Recipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Difficulty: {recipe.difficulty}")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
print("Instructions:")
for i, step in enumerate(recipe.instructions, 1):
    print(f"  {i}. {step}")
```

##### Structured Output with Error Handling
```python
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat
from pydantic import BaseModel, Field
from typing import Optional

chat_model = SecureGPTChat()

class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

# Use include_raw=True for better error handling
structured_llm = chat_model.with_structured_output(Joke, include_raw=True)

messages = [HumanMessage(content="Tell me a joke about programming")]
result = structured_llm.invoke(messages)

if result.get('parsing_error'):
    print(f"Parsing error occurred: {result['parsing_error']}")
    print(f"Raw response: {result['raw'].content}")
else:
    joke = result['parsed']
    print(f"Setup: {joke.setup}")
    print(f"Punchline: {joke.punchline}")
```

##### Streaming with Structured Output
```python
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat
from pydantic import BaseModel, Field

chat_model = SecureGPTChat()

class Story(BaseModel):
    """A short story."""
    title: str = Field(description="Title of the story")
    genre: str = Field(description="Genre of the story")
    characters: list[str] = Field(description="Main characters in the story")
    plot: str = Field(description="The main plot of the story")
    word_count: int = Field(description="Approximate word count")

structured_llm = chat_model.with_structured_output(Story)
messages = [HumanMessage(content="Write a short fantasy story about a magical library")]

print("Streaming story generation...")
for chunk in structured_llm.stream(messages):
    if hasattr(chunk, 'content'):
        # Raw streaming chunks
        print(chunk.content, end='', flush=True)
    elif isinstance(chunk, Story):
        # Final structured output
        print(f"\n\nStructured output received:")
        print(f"  Title: {chunk.title}")
        print(f"  Genre: {chunk.genre}")
        print(f"  Characters: {', '.join(chunk.characters)}")
        print(f"  Word count: {chunk.word_count}")
        break
```

#### Tool Calling Examples

The SecureGPTChat class supports tool calling functionality, allowing the model to execute functions and use external tools. This enables more complex workflows where the model can perform actions beyond just generating text.

##### Basic Tool Creation with @tool Decorator

```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import json

# Define a simple tool using the @tool decorator
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather information for a specified location.

    Args:
        location: The city and state/country for weather lookup
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Simulate weather API call
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 10
    }
    return json.dumps(weather_data)

@tool
def calculate(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a * b

# Initialize chat model and bind tools
chat_model = SecureGPTChat(temperature=0.1)
llm_with_tools = chat_model.bind_tools([get_weather, calculate])

# Use the tools
message = HumanMessage(content="What's the weather like in Tokyo? Also, what's 15 * 23?")
response = llm_with_tools.invoke([message])

print(f"Response: {response.content}")

# Check if tool calls were made
if hasattr(response, 'tool_calls') and response.tool_calls:
    print(f"Tool calls made: {len(response.tool_calls)}")
    for tool_call in response.tool_calls:
        print(f"- Tool: {tool_call['name']}")
        print(f"- Arguments: {tool_call['args']}")
```

##### Advanced Tool Creation with BaseTool

```python
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from typing import Optional
import json
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="The city and state/country for weather lookup")
    unit: str = Field(default="celsius", description="Temperature unit (celsius or fahrenheit)")

class WeatherTool(BaseTool):
    """Custom weather tool that simulates weather API calls."""
    name: str = "get_weather_info"
    description: str = "Get current weather information for a specified location"
    args_schema: Optional[type[BaseModel]] = WeatherInput
    return_direct: bool = False

    def _run(
        self,
        location: str,
        unit: str = "celsius",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the weather tool."""
        weather_data = {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 10
        }
        return json.dumps(weather_data)

    async def _arun(
        self,
        location: str,
        unit: str = "celsius",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version of the weather tool."""
        return self._run(location, unit)

# Use the custom tool
weather_tool = WeatherTool()
chat_model = SecureGPTChat(temperature=0.1)
llm_with_tools = chat_model.bind_tools([weather_tool])

message = HumanMessage(content="What's the weather in London?")
response = llm_with_tools.invoke([message])
print(f"Response: {response.content}")
```

##### Tool Choice Control

```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

chat_model = SecureGPTChat(temperature=0.1)

# Different tool choice strategies
message = HumanMessage(content="What's the weather in London?")

# Let the model decide when to use tools (default)
llm_auto = chat_model.bind_tools([get_weather], tool_choice="auto")
response_auto = llm_auto.invoke([message])

# Don't use any tools
llm_none = chat_model.bind_tools([get_weather], tool_choice="none")
response_none = llm_none.invoke([message])

# Force specific tool usage
llm_force = chat_model.bind_tools([get_weather], tool_choice={
    "type": "function", 
    "function": {"name": "get_weather"}
})
response_force = llm_force.invoke([message])

print(f"Auto choice: {response_auto.content}")
print(f"No tools: {response_none.content}")
print(f"Forced tool: {response_force.content}")
```

##### Tool Response Handling

```python
from langchain_core.messages import HumanMessage, ToolMessage
from cai_securegpt_client import SecureGPTChat


chat_model = SecureGPTChat(temperature=0.1)
llm_with_tools = chat_model.bind_tools([get_weather])

# Start conversation
messages = [HumanMessage(content="What's the weather like in San Francisco?")]

# First call - should trigger tool use
response = llm_with_tools.invoke(messages)
messages.append(response)

# If tool calls were made, simulate tool responses
if hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
        # Execute the tool
        if tool_call['name'] == "get_weather":
            tool_result = get_weather.invoke(tool_call['args'])
            
            # Add tool response to conversation
            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call['id']
            )
            messages.append(tool_message)
    
    # Continue conversation with tool results
    final_response = llm_with_tools.invoke(messages)
    print(f"Final response: {final_response.content}")
```

##### Async Tool Calling

```python
import asyncio
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat


@tool
async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def async_tool_example():
    chat_model = SecureGPTChat(temperature=0.1)
    llm_with_tools = chat_model.bind_tools([get_current_time, calculate])
    
    message = HumanMessage(content="What time is it now? Also calculate 123 * 456")
    response = await llm_with_tools.ainvoke([message])
    
    print(f"Async response: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"- Tool: {tool_call['name']}")
            print(f"- Arguments: {tool_call['args']}")

# Run async example
asyncio.run(async_tool_example())
```

##### Error Handling in Tools

```python
from langchain_core.tools import ToolException, StructuredTool

def faulty_weather(city: str) -> str:
    """Get weather for a city that always fails."""
    raise ToolException(f"Error: There is no city by the name of {city}.")

# Create tool with error handling
error_tool = StructuredTool.from_function(
    func=faulty_weather,
    name="faulty_weather",
    description="A weather tool that always fails",
    handle_tool_error=True  # This will catch ToolException
)

# Test tool with error handling
try:
    result = error_tool.invoke({"city": "NonExistentCity"})
    print(f"Tool result with error handling: {result}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Custom error handler
def custom_error_handler(error: ToolException) -> str:
    return f"Custom error message: {error.args[0]}"

error_tool_custom = StructuredTool.from_function(
    func=faulty_weather,
    name="faulty_weather_custom",
    description="A weather tool with custom error handling",
    handle_tool_error=custom_error_handler
)

result = error_tool_custom.invoke({"city": "AnotherFakeCity"})
print(f"Tool result with custom error handler: {result}")
```

##### Tools with Type Annotations

```python
from typing import List, Annotated
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from cai_securegpt_client import SecureGPTChat

@tool
def multiply_by_max(
    a: Annotated[int, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)

chat_model = SecureGPTChat(temperature=0.1)
llm_with_tools = chat_model.bind_tools([multiply_by_max])

message = HumanMessage(content="Multiply 5 by the maximum of [10, 20, 15, 8]")
response = llm_with_tools.invoke([message])
print(f"Response: {response.content}")
```

#### Embeddings Examples

##### Basic Embeddings
```python
from cai_securegpt_client import SecureGptEmbeddings

# Initialize embeddings
embeddings = SecureGptEmbeddings()

# Get embedding for a single query
result = embeddings.embed_query("Hello, how are you?")
print(f"Embedding dimensions: {len(result)}")
print(f"First 5 values: {result[:5]}")
```

##### Multiple Document Embeddings
```python
from cai_securegpt_client import SecureGptEmbeddings

# Initialize embeddings
embeddings = SecureGptEmbeddings()

# Embed multiple documents
documents = [
    "The weather is nice today.",
    "I love programming in Python.",
    "Machine learning is fascinating."
]

doc_embeddings = embeddings.embed_documents(documents)
print(f"Generated {len(doc_embeddings)} embeddings")
print(f"Each embedding has {len(doc_embeddings[0])} dimensions")
```

##### Async Embeddings
```python
import asyncio
from cai_securegpt_client import SecureGptEmbeddings

# Initialize embeddings
embeddings = SecureGptEmbeddings()


async def async_embeddings():
    # Async single query
    result = await embeddings.aembed_query("Hello, how are you?")
    print(f"Async embedding result: {len(result)} dimensions")
    
    # Async multiple documents
    documents = ["Document 1", "Document 2", "Document 3"]
    doc_results = await embeddings.aembed_documents(documents)
    print(f"Async doc embeddings: {len(doc_results)} results")

asyncio.run(async_embeddings())
```

#### Multimodal Examples (Text + Images)

The SecureGPTChat class supports multimodal inputs, allowing you to send both text and images to vision-enabled models like GPT-4 Vision.

##### Basic Image Analysis
```python
import base64
from cai_securegpt_client import SecureGPTChat

# Initialize the chat model with a vision-capable model
chat_model = SecureGPTChat(
    model_id="gpt-4o",  # Use a vision-capable model
    temperature=0.7
)

# Method 1: Using the convenience helper method
def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load and encode your image
image_base64 = encode_image_to_base64("path/to/your/image.jpg")

# Create multimodal message using helper method
message = chat_model.create_multimodal_message(
    text="What do you see in this image? Describe it in detail.",
    image_base64=image_base64,
    detail="auto"  # Options: "auto", "low", "high"
)

response = chat_model.invoke([message])
print(f"Image analysis: {response.content}")
```

##### Raw Dictionary Format (Direct API Style)
```python
from cai_securegpt_client import SecureGPTChat

chat_model = SecureGPTChat(model_id="gpt-4o")

# Method 2: Using raw dictionary format (matches OpenAI API directly)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "source_type": "base64",
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
                "detail": "auto"
            },
            {
                "type": "text",
                "text": "What are the main objects in this image?"
            }
        ]
    }
]

# The invoke method now accepts raw dictionary messages
response = chat_model.invoke(messages)
print(f"Analysis: {response.content}")
```

##### Multi-Image Comparison
```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

chat_model = SecureGPTChat(model_id="gpt-4o")

# Compare multiple images in a single request
multimodal_content = [
    {
        "type": "text",
        "text": "Compare these two images and tell me the differences:"
    },
    {
        "type": "image_url",
        "image_url": {
            "source_type": "base64",
            "url": f"data:image/jpeg;base64,{first_image_base64}"
        },
        "detail": "high"
    },
    {
        "type": "image_url",
        "image_url": {
            "source_type": "base64",
            "url": f"data:image/jpeg;base64,{second_image_base64}"
        },
        "detail": "high"
    }
]

message = HumanMessage(content=multimodal_content)
response = chat_model.invoke([message])
print(f"Comparison: {response.content}")
```

##### Streaming Multimodal Responses
```python
from cai_securegpt_client import SecureGPTChat

chat_model = SecureGPTChat(model_id="gpt-4o")

# Create multimodal message
message = chat_model.create_multimodal_message(
    text="Describe this image step by step, focusing on colors, objects, and composition.",
    image_base64=image_base64,
    detail="high"
)

print("Streaming image analysis:")
for chunk in chat_model.stream([message]):
    if chunk.content:
        print(chunk.content, end='', flush=True)
print()  # New line after streaming
```

##### Async Multimodal Operations
```python
import asyncio
from cai_securegpt_client import SecureGPTChat

async def analyze_image_async():
    chat_model = SecureGPTChat(model_id="gpt-4o")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "source_type": "base64",
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                    "detail": "auto"
                },
                {
                    "type": "text",
                    "text": "Extract all the text visible in this image."
                }
            ]
        }
    ]
    
    response = await chat_model.ainvoke(messages)
    print(f"Extracted text: {response.content}")

# Run async example
asyncio.run(analyze_image_async())
```

##### Multimodal with System Instructions
```python
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import SystemMessage

chat_model = SecureGPTChat(model_id="gpt-4o")

# Mix LangChain message objects with raw dictionary multimodal messages
messages = [
    SystemMessage(content="You are an expert art critic. Analyze images with professional insight."),
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please provide a detailed art critique of this painting:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "source_type": "base64",
                    "url": f"data:image/jpeg;base64,{painting_image_base64}"
                },
                "detail": "high"
            }
        ]
    }
]

response = chat_model.invoke(messages)
print(f"Art critique: {response.content}")
```

##### Image Detail Levels
```python
from cai_securegpt_client import SecureGPTChat

chat_model = SecureGPTChat(model_id="gpt-4o")

# Different detail levels affect processing speed and cost
detail_levels = ["low", "auto", "high"]

for detail in detail_levels:
    message = chat_model.create_multimodal_message(
        text=f"Analyze this image with {detail} detail level.",
        image_base64=image_base64,
        detail=detail
    )
    
    response = chat_model.invoke([message])
    print(f"\n{detail.upper()} detail analysis:")
    print(response.content[:200] + "...")  # Show first 200 chars
```

### Configuration

You need to configure certain parameters based on your environment. Configuration options include:

 * Model and provider selection
 * Temperature
 * Timeout and retries
 * API keys for OneLogin

Configuration can be done through a _.env_ configuration file or environment variables. You can instead set the parameters when instantiating the *SecureGPTChat* or *SecureGptEmbeddings* objects.

#### Environment variables
```bash
SECUREGPT_ONELOGIN_CLIENT_ID={client_id} # (Mandatory)
SECUREGPT_ONELOGIN_SECRET={client_secret} # (Mandatory)
SECUREGPT_ONELOGIN_URL=https://onelogin.axa.com/as/token.oauth2 # (Optional) Defaults to https://onelogin.axa.com/as/token.oauth2 if not provided
SECUREGPT_MODEL_ID=gpt-4o-mini-2024-07-18 # (Optional) Defaults to gpt-4o-mini-2024-07-18 if not provided
SECUREGPT_PROVIDER=openai # (Optional) Defaults to openai if not provided
SECUREGPT_OPENAI_API_VERSION=2024-10-21 # (Optional) Defaults to 2024-10-21 if not provided
SECUREGPT_URL=https://api.se.axa-go.applications.services.axa-tech.intraxa/ago-m365-securegpt-hub-v1-vrs # (Optional) Defaults to https://api.se.axa-go.applications.services.axa-tech.intraxa/ago-m365-securegpt-hub-v1-vrs if not provided
SECUREGPT_TEMPERATURE=0.7 # (Optional) Defaults to 0.7 if not provided
SECUREGPT_TIMEOUT=120 # (Optional) Defaults to 120 if not provided
SECUREGPT_MAX_RETRIES=2 # (Optional) Defaults to 2 if not provided
SECUREGPT_EMBEDDINGS_MODEL_ID=text-embedding-ada-002-2 # (Optional) Defaults to text-embedding-ada-002-2 if not provided
SECUREGPT_EMBEDDINGS_PROVIDER=openai # (Optional) Defaults to openai if not provided
```

### Example Files

The `examples/` directory contains complete working examples:

- **`simple_chat_example.py`** - Demonstrates basic chat functionality, multi-turn conversations, streaming, async operations, and different temperature settings
- **`structured_output_example.py`** - Shows how to use Pydantic models for structured output generation with error handling and streaming
- **`multimodal_example.py`** - Illustrates how to use the chat model with images, including basic image analysis, multi-image comparison, and streaming responses
- **`tool_calling_example.py`** - Demonstrates how to use the chat model with tool calling capabilities, including structured output and error handling

Run the examples with:
```bash
cd examples
python simple_chat_example.py
python structured_output_example.py
python multimodal_example.py
python tool_calling_example.py
```

### Best Practices

- Use temperature 0.1-0.3 for factual, consistent responses
- Use temperature 0.7-0.9 for creative, varied responses
- System messages help control the assistant's behavior
- Streaming is useful for real-time user interfaces
- Async operations are great for handling multiple requests
- Always use BaseMessage objects (HumanMessage, SystemMessage, etc.)
- Use `include_raw=True` for better error handling with structured output
- Define clear Pydantic models with Field descriptions
- Handle parsing errors gracefully in production code

### Contributing

We welcome all contributions. To contribute to the project:

 1. Fork the repository.
 2. Create a new branch (git checkout -b feature-branch).
 3. Make your changes and commit them (git commit -m 'Add new feature').
 4. Push to your branch (git push origin feature-branch).
 5. Create a pull request.

Please ensure your code follows the coding standards and includes appropriate tests.

### Contact

For questions or feedback, please reach out emilio.manzano@axa.com.

----------------------------------------------------------------------------------------------------------------------------------