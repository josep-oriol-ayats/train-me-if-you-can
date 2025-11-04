#!/usr/bin/env python3
"""
Tool calling example script demonstrating function calling capabilities with SecureGPTChat.

This script shows how to use the SecureGPTChat model for:
1. Using the @tool decorator (recommended)
2. Using StructuredTool.from_function
3. Subclassing BaseTool (most flexible)
4. Handling tool responses
5. Multi-step tool calling workflows
6. Error handling for tool calls
7. Async tool calling
"""

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool
from pydantic import BaseModel, Field

from cai_securegpt_client import SecureGPTChat

load_dotenv()


# Define Pydantic models for structured tool inputs
class WeatherInput(BaseModel):
    """Input for weather queries."""

    location: str = Field(description="The city and state/country for weather lookup")
    unit: str = Field(
        default="celsius",
        description="Temperature unit (celsius or fahrenheit)",
    )


class CalculatorInput(BaseModel):
    """Input for calculator operations."""

    a: int = Field(description="first number")
    b: int = Field(description="second number")


class SearchInput(BaseModel):
    """Input for search queries."""

    query: str = Field(description="Search query string")
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
    )


# Method 1: Using @tool decorator (Recommended)
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
        "wind_speed": 10,
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


@tool
async def get_current_time() -> str:
    """Get the current time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@tool
def reverse_string(text: str) -> str:
    """Reverse a string.

    Args:
        text: The string to reverse
    """
    return text[::-1]


# Method 2: Using StructuredTool.from_function
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers asynchronously."""
    return a * b


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    # Simulate web search
    results = [f"Result {i + 1}: Information about {query}" for i in range(min(max_results, 3))]
    return json.dumps({"query": query, "results": results})


# Method 3: Subclassing BaseTool (Most flexible)
class WeatherTool(BaseTool):
    """Custom weather tool that simulates weather API calls."""

    name: str = "get_weather_info"
    description: str = "Get current weather information for a specified location"
    args_schema: type[BaseModel] | None = WeatherInput
    return_direct: bool = False

    def _run(
        self,
        location: str,
        unit: str = "celsius",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the weather tool."""
        # Simulate weather API call
        weather_data = {
            "location": location,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 10,
        }
        return json.dumps(weather_data)

    async def _arun(
        self,
        location: str,
        unit: str = "celsius",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Async version of the weather tool."""
        # For simple operations, delegate to sync version
        return self._run(
            location,
            unit,
            run_manager=run_manager.get_sync() if run_manager else None,
        )


class AdvancedCalculatorTool(BaseTool):
    """Advanced calculator tool with multiple operations."""

    name: str = "advanced_calculator"
    description: str = "Perform various mathematical operations"
    args_schema: type[BaseModel] | None = CalculatorInput

    def _run(
        self,
        a: int,
        b: int,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the calculator tool."""
        try:
            result = {
                "addition": a + b,
                "subtraction": a - b,
                "multiplication": a * b,
                "division": a / b if b != 0 else "undefined",
            }
            return json.dumps(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except (TypeError, ValueError) as e:
            return f"Error: {e!r}"

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Async version of the calculator tool."""
        return self._run(
            a,
            b,
            run_manager=run_manager.get_sync() if run_manager else None,
        )


def demonstrate_tool_decorator() -> None:
    """Demonstrate using @tool decorator."""
    print("=== @tool Decorator Example ===")

    chat_model = SecureGPTChat(temperature=0.1)

    # Bind tools created with @tool decorator
    llm_with_tools = chat_model.bind_tools([get_weather, calculate])

    message = HumanMessage(
        content="What's the weather like in Tokyo? Also, what's 15 * 23?",
    )

    try:
        response = llm_with_tools.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        # Check if tool calls were made - access as attribute first
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                # Tool calls are ToolCall objects with name, args, id keys
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")
                print(f"- ID: {tool_call['id']}")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("This might be expected in demo mode if authentication is not set up.")


def demonstrate_structured_tool() -> None:
    """Demonstrate using StructuredTool.from_function."""
    print("\n=== StructuredTool.from_function Example ===")

    chat_model = SecureGPTChat(temperature=0.1)

    # Create tools using StructuredTool.from_function
    calculator_tool = StructuredTool.from_function(
        func=multiply_numbers,
        name="Calculator",
        description="Multiply two numbers",
        args_schema=CalculatorInput,
        coroutine=amultiply_numbers,  # Async version
    )

    search_tool = StructuredTool.from_function(
        func=search_web,
        name="WebSearch",
        description="Search the web for information",
        args_schema=SearchInput,
    )

    llm_with_tools = chat_model.bind_tools([calculator_tool, search_tool])

    message = HumanMessage(content="Calculate 45 * 67 and search for 'LangChain tools'")

    try:
        response = llm_with_tools.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")

    except Exception as e:
        print(f"Error occurred: {e}")


def demonstrate_base_tool_subclass() -> None:
    """Demonstrate subclassing BaseTool."""
    print("\n=== BaseTool Subclass Example ===")

    chat_model = SecureGPTChat(temperature=0.1)

    # Create instances of custom tools
    weather_tool = WeatherTool()
    calculator_tool = AdvancedCalculatorTool()

    llm_with_tools = chat_model.bind_tools([weather_tool, calculator_tool])

    message = HumanMessage(
        content="What's the weather in London and calculate operations for 12 and 4?",
    )

    try:
        response = llm_with_tools.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")

    except Exception as e:
        print(f"Error occurred: {e}")


def demonstrate_tool_choice_control() -> None:
    """Demonstrate controlling tool choice behavior."""
    print("\n=== Tool Choice Control Example ===")

    chat_model = SecureGPTChat(temperature=0.1)

    # Test different tool choice strategies
    tool_choice_examples = [
        ("auto", "Let the model decide when to use tools"),
        ("none", "Don't use any tools"),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            "Force weather tool",
        ),
    ]

    message = HumanMessage(content="What's the weather in London?")

    for tool_choice, description in tool_choice_examples:
        print(f"\nTool Choice: {description}")
        try:
            llm_with_tools = chat_model.bind_tools(
                [get_weather, calculate],
                tool_choice=tool_choice,  # type: ignore
            )

            response = llm_with_tools.invoke([message])
            print(f"Response: {response.content}")

            if hasattr(response, "tool_calls") and response.tool_calls:
                print(f"Tools called: {[tc['name'] for tc in response.tool_calls]}")
            else:
                print("No tools called")

        except Exception as e:
            print(f"Error: {e}")


def demonstrate_tool_response_handling() -> None:
    """Demonstrate handling tool responses in a conversation."""
    print("\n=== Tool Response Handling Example ===")

    chat_model = SecureGPTChat(temperature=0.1)
    llm_with_tools = chat_model.bind_tools([get_weather])

    # Start conversation
    messages: List[BaseMessage] = [HumanMessage(content="What's the weather like in San Francisco?")]

    try:
        # First call - should trigger tool use
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        print(f"User: {messages[0].content}")
        print(f"Assistant: {response.content}")

        # If tool calls were made, simulate tool responses
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")

            for tool_call in response.tool_calls:
                print(f"Executing tool: {tool_call['name']}")

                # Execute the tool
                if tool_call["name"] == "get_weather":
                    tool_result = get_weather.invoke(tool_call["args"])
                    print(f"Tool result: {tool_result}")

                    # Add tool response to conversation
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                    messages.append(tool_message)

            # Continue conversation with tool results
            final_response = llm_with_tools.invoke(messages)
            print(f"Final response: {final_response.content}")

    except Exception as e:
        print(f"Error occurred: {e}")


async def demonstrate_async_tool_calling() -> None:
    """Demonstrate async tool calling."""
    print("\n=== Async Tool Calling Example ===")

    chat_model = SecureGPTChat(temperature=0.1)
    llm_with_tools = chat_model.bind_tools([get_current_time, calculate])

    message = HumanMessage(content="What time is it now? Also calculate 123 * 456")

    try:
        response = await llm_with_tools.ainvoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")

    except Exception as e:
        print(f"Error occurred: {e}")


def demonstrate_tool_with_annotations() -> None:
    """Demonstrate tools with type annotations."""
    print("\n=== Tools with Type Annotations Example ===")

    @tool
    def multiply_by_max(
        a: Annotated[int, "scale factor"],
        b: Annotated[list[int], "list of ints over which to take maximum"],
    ) -> int:
        """Multiply a by the maximum of b."""
        return a * max(b)

    chat_model = SecureGPTChat(temperature=0.1)
    llm_with_tools = chat_model.bind_tools([multiply_by_max])

    message = HumanMessage(content="Multiply 10 by the maximum of [1, 3, 5, 4, 2]")

    try:
        response = llm_with_tools.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")

    except Exception as e:
        print(f"Error occurred: {e}")


def demonstrate_tool_with_artifacts() -> None:
    """Demonstrate tools that return artifacts."""
    print("\n=== Tools with Artifacts Example ===")

    @tool(response_format="content_and_artifact")
    def generate_random_numbers(
        min_val: int,
        max_val: int,
        count: int,
    ) -> tuple[str, list[int]]:
        """Generate random numbers in a specified range.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            count: Number of values to generate
        """
        numbers = [random.randint(min_val, max_val) for _ in range(count)]
        content = f"Generated {count} random numbers between {min_val} and {max_val}"
        return content, numbers

    chat_model = SecureGPTChat(temperature=0.1)
    llm_with_tools = chat_model.bind_tools([generate_random_numbers])

    message = HumanMessage(content="Generate 5 random numbers between 1 and 100")

    try:
        response = llm_with_tools.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"- Tool: {tool_call['name']}")
                print(f"- Arguments: {tool_call['args']}")

                # Execute tool to see artifact
                tool_result = generate_random_numbers.invoke(
                    {
                        "name": "generate_random_numbers",
                        "args": tool_call["args"],
                        "id": tool_call["id"],
                        "type": "tool_call",
                    }
                )
                print(f"- Tool result: {tool_result}")

    except Exception as e:
        print(f"Error occurred: {e}")


def demonstrate_tool_error_handling() -> None:
    """Demonstrate error handling in tool calling."""
    print("\n=== Tool Error Handling Example ===")

    def faulty_weather(city: str) -> str:
        """Get weather for a city that always fails."""
        raise ToolException(f"Error: There is no city by the name of {city}.")

    # Create tool with error handling
    error_tool = StructuredTool.from_function(
        func=faulty_weather,
        name="faulty_weather",
        description="A weather tool that always fails",
        handle_tool_error=True,  # This will catch ToolException
    )

    # Test direct tool invocation
    try:
        result = error_tool.invoke({"city": "NonExistentCity"})
        print(f"Tool result with error handling: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test with custom error handler
    def custom_error_handler(error: ToolException) -> str:
        return f"Custom error message: {error.args[0]}"

    error_tool_custom = StructuredTool.from_function(
        func=faulty_weather,
        name="faulty_weather_custom",
        description="A weather tool with custom error handling",
        handle_tool_error=custom_error_handler,
    )

    try:
        result = error_tool_custom.invoke({"city": "AnotherFakeCity"})
        print(f"Tool result with custom error handler: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main() -> None:
    """Run all tool calling examples."""
    print("SecureGPT Tool Calling Examples")
    print("=" * 50)

    # Run synchronous examples
    demonstrate_tool_decorator()
    demonstrate_structured_tool()
    demonstrate_base_tool_subclass()
    demonstrate_tool_choice_control()
    demonstrate_tool_response_handling()
    demonstrate_tool_with_annotations()
    demonstrate_tool_with_artifacts()
    demonstrate_tool_error_handling()

    # Run async example
    print("\n" + "=" * 50)
    print("Running async examples...")
    asyncio.run(demonstrate_async_tool_calling())

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
