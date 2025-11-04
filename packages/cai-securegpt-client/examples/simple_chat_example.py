#!/usr/bin/env python3
"""
Simple example script demonstrating basic LLM calls and chat calls with SecureGPTChat.

This script shows how to use the SecureGPTChat model for:
1. Basic single message calls
2. Multi-turn conversations
3. Streaming responses
4. Async operations
5. Different temperature settings
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cai_securegpt_client import SecureGPTChat

load_dotenv()


def demonstrate_basic_call():
    """Demonstrate basic single message call."""
    print("=== Basic LLM Call Example ===")

    # Initialize the chat model
    chat_model = SecureGPTChat(temperature=0.7, max_retries=2, timeout=60)

    # Simple message
    message = HumanMessage(content="Tell me a short joke about programming.")

    try:
        # Make the call
        response = chat_model.invoke([message])
        print(f"Question: {message.content}")
        print(f"Response: {response.content}")
        print(f"Response type: {type(response)}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("This might be expected in demo mode if authentication is not set up.")


def demonstrate_multi_turn_conversation():
    """Demonstrate multi-turn conversation."""
    print("\n=== Multi-turn Conversation Example ===")

    chat_model = SecureGPTChat(temperature=0.5)

    # Start with a system message to set context
    messages = [
        SystemMessage(content="You are a helpful assistant that gives concise, friendly responses."),
        HumanMessage(content="Hi! Can you explain what Python is?"),
    ]

    try:
        # First exchange
        response1 = chat_model.invoke(messages)
        print(f"User: {messages[1].content}")
        print(f"Assistant: {response1.content}")

        # Add the assistant's response and continue the conversation
        messages.append(AIMessage(content=response1.content))
        messages.append(HumanMessage(content="What are some popular Python frameworks?"))

        # Second exchange
        response2 = chat_model.invoke(messages)
        print(f"\nUser: {messages[3].content}")
        print(f"Assistant: {response2.content}")

        # Show the complete conversation history
        print(f"\nTotal messages in conversation: {len(messages) + 1}")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Expected behavior:")
        print("1. First response would explain Python")
        print("2. Second response would list popular frameworks like Django, Flask, FastAPI")


def demonstrate_streaming():
    """Demonstrate streaming responses."""
    print("\n=== Streaming Response Example ===")

    chat_model = SecureGPTChat(temperature=0.8)

    message = HumanMessage(content="Write a short story about a robot learning to cook. Keep it under 200 words.")

    try:
        print("Question:", message.content)
        print("Streaming response:")
        print("-" * 50)

        # Stream the response
        full_response = ""
        for chunk in chat_model.stream([message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

        print("\n" + "-" * 50)
        print(f"Complete response length: {len(full_response)} characters")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Expected streaming behavior:")
        print("- Text appears word by word in real-time")
        print("- Each chunk contains a small piece of the response")
        print("- Final response is built up incrementally")


async def demonstrate_async_operations():
    """Demonstrate asynchronous operations."""
    print("\n=== Async Operations Example ===")

    chat_model = SecureGPTChat(temperature=0.6)

    # Single async call
    message = HumanMessage(content="What are the benefits of asynchronous programming?")

    try:
        print("Making async call...")
        response = await chat_model.ainvoke([message])
        print(f"Async response: {response.content}")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Expected: Explanation of async programming benefits")


async def demonstrate_async_streaming():
    """Demonstrate asynchronous streaming."""
    print("\n=== Async Streaming Example ===")

    chat_model = SecureGPTChat(temperature=0.7)

    message = HumanMessage(content="Explain the concept of machine learning in simple terms.")

    try:
        print("Question:", message.content)
        print("Async streaming response:")
        print("-" * 50)

        full_response = ""
        async for chunk in chat_model.astream([message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

        print("\n" + "-" * 50)
        print(f"Complete response length: {len(full_response)} characters")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Expected: Streamed explanation of machine learning")


def demonstrate_different_temperatures():
    """Demonstrate how temperature affects responses."""
    print("\n=== Temperature Comparison Example ===")

    prompt = "Complete this sentence: The future of artificial intelligence is"
    message = HumanMessage(content=prompt)

    temperatures = [0.1, 0.5, 0.9]

    for temp in temperatures:
        chat_model = SecureGPTChat(temperature=temp)

        try:
            response = chat_model.invoke([message])
            print(f"Temperature {temp}: {response.content}")
        except Exception as e:
            print(f"Temperature {temp}: Error - {e}")
            print(f"Expected: {'More deterministic' if temp < 0.5 else 'More creative'} response")


def demonstrate_system_prompts():
    """Demonstrate using system prompts to control behavior."""
    print("\n=== System Prompt Example ===")

    chat_model = SecureGPTChat(temperature=0.7)

    # Different system prompts for different behaviors
    system_prompts = [
        "You are a helpful assistant that speaks like a pirate.",
        "You are a technical expert who gives very detailed explanations.",
        "You are a poet who responds only in rhyming verse.",
    ]

    user_message = "Tell me about the ocean."

    for i, system_prompt in enumerate(system_prompts, 1):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            response = chat_model.invoke(messages)
            print(f"Style {i} ({system_prompt[:20]}...): {response.content}")
        except Exception as e:
            print(f"Style {i}: Error - {e}")


async def run_async_examples():
    """Run all async examples."""
    await demonstrate_async_operations()
    await demonstrate_async_streaming()


def main():
    """Run all examples."""
    print("SecureGPTChat Simple Examples")
    print("=" * 50)

    # Synchronous examples
    demonstrate_basic_call()
    demonstrate_multi_turn_conversation()
    demonstrate_streaming()
    demonstrate_different_temperatures()
    demonstrate_system_prompts()

    # Asynchronous examples
    print("\n" + "=" * 50)
    print("Running async examples...")
    asyncio.run(run_async_examples())

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTips for using SecureGPTChat:")
    print("- Use temperature 0.1-0.3 for factual, consistent responses")
    print("- Use temperature 0.7-0.9 for creative, varied responses")
    print("- System messages help control the assistant's behavior")
    print("- Streaming is useful for real-time user interfaces")
    print("- Async operations are great for handling multiple requests")


if __name__ == "__main__":
    main()
