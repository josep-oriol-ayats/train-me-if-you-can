#!/usr/bin/env python3
"""
Example script demonstrating structured output generation with SecureGPTChat.

This script shows how to use the SecureGPTChat model to generate structured output
using Pydantic models, with proper message formatting and error handling.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from cai_securegpt_client import SecureGPTChat

load_dotenv()


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(default=None, description="How funny the joke is, from 1 to 10")


class Recipe(BaseModel):
    """A cooking recipe."""

    name: str = Field(description="Name of the dish")
    ingredients: list[str] = Field(description="List of ingredients needed")
    instructions: list[str] = Field(description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")


class PersonInfo(BaseModel):
    """Information about a person."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")
    hobbies: list[str] = Field(description="List of hobbies or interests")
    location: str = Field(description="City or country where they live")


class Story(BaseModel):
    """A short story."""

    title: str = Field(description="Title of the story")
    genre: str = Field(description="Genre of the story")
    characters: list[str] = Field(description="Main characters in the story")
    plot: str = Field(description="The main plot of the story")
    word_count: int = Field(description="Approximate word count")


class NewsArticle(BaseModel):
    """A news article."""

    headline: str = Field(description="The headline of the article")
    summary: str = Field(description="Brief summary of the article")
    key_points: list[str] = Field(description="Key points or highlights")
    category: str = Field(description="News category (politics, sports, tech, etc.)")
    urgency: str = Field(description="Urgency level: low, medium, high")


def setup_environment():
    """Set up environment variables for testing."""
    # Set up test environment variables if not already set
    if not os.getenv("SECUREGPT_URL"):
        os.environ["SECUREGPT_URL"] = "https://test.example.com"
    if not os.getenv("SECUREGPT_MODEL_ID"):
        os.environ["SECUREGPT_MODEL_ID"] = "gpt-4"
    if not os.getenv("SECUREGPT_PROVIDER"):
        os.environ["SECUREGPT_PROVIDER"] = "openai"


def mock_auth_for_demo():
    """Set up mock authentication for demo purposes."""
    # In a real scenario, you would set up proper authentication
    # For this demo, we'll use a placeholder token
    os.environ["SECUREGPT_ACCESS_TOKEN"] = "demo_token_placeholder"


def demonstrate_joke_generation():
    """Demonstrate generating a structured joke."""
    print("=== Joke Generation Example ===")

    # Create the chat model
    chat_model = SecureGPTChat()

    # Create structured output model
    structured_llm = chat_model.with_structured_output(Joke)

    # Create proper message format (this is the key fix!)
    messages = [HumanMessage(content="Tell me a joke about cats")]

    try:
        # Note: This will fail in demo mode without real API credentials
        # But shows the correct usage pattern
        answer = structured_llm.invoke(messages)
        print(f"Setup: {answer.setup}")
        print(f"Punchline: {answer.punchline}")
        print(f"Rating: {answer.rating}/10")
    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_recipe_generation():
    """Demonstrate generating a structured recipe."""
    print("\n=== Recipe Generation Example ===")

    chat_model = SecureGPTChat()

    structured_llm = chat_model.with_structured_output(Recipe)

    # Proper message format
    messages = [HumanMessage(content="Give me a simple recipe for chocolate chip cookies")]

    try:
        recipe = structured_llm.invoke(messages)
        print(f"Recipe: {recipe.name}")
        print(f"Prep time: {recipe.prep_time_minutes} minutes")
        print(f"Difficulty: {recipe.difficulty}")
        print(f"Ingredients: {', '.join(recipe.ingredients)}")
        print("Instructions:")
        for i, step in enumerate(recipe.instructions, 1):
            print(f"  {i}. {step}")
    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_with_include_raw():
    """Demonstrate structured output with raw response included."""
    print("\n=== Structured Output with Raw Response ===")

    chat_model = SecureGPTChat()

    # Use include_raw=True to get both parsed and raw response
    structured_llm = chat_model.with_structured_output(PersonInfo, include_raw=True)

    messages = [HumanMessage(content="Create a fictional character profile for a fantasy story")]

    try:
        result = structured_llm.invoke(messages)
        print("Parsed data:")
        print(f"  Name: {result['parsed'].name}")
        print(f"  Age: {result['parsed'].age}")
        print(f"  Occupation: {result['parsed'].occupation}")
        print(f"  Location: {result['parsed'].location}")
        print(f"  Hobbies: {', '.join(result['parsed'].hobbies)}")
        print(f"\nRaw response available: {result['raw'] is not None}")
    except Exception as e:
        print(f"Error: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling with structured output."""
    print("\n=== Error Handling Example ===")

    chat_model = SecureGPTChat()

    # Use include_raw=True for better error handling
    structured_llm = chat_model.with_structured_output(Joke, include_raw=True)

    messages = [HumanMessage(content="Tell me something that's not a joke")]

    try:
        result = structured_llm.invoke(messages)

        if result.get("parsing_error"):
            print(f"Parsing error occurred: {result['parsing_error']}")
            print(f"Raw response: {result['raw'].content}")
        else:
            print("Successfully parsed structured output")
    except Exception as e:
        print(f"Demo mode - would demonstrate error handling. Error: {e}")


async def demonstrate_async_usage():
    """Demonstrate async structured output generation."""
    print("\n=== Async Usage Example ===")

    chat_model = SecureGPTChat()

    structured_llm = chat_model.with_structured_output(Joke)

    messages = [HumanMessage(content="Tell me a joke about programming")]

    try:
        # Async invoke
        joke = await structured_llm.ainvoke(messages)
        print(f"Async joke - Setup: {joke.setup}")
        print(f"Async joke - Punchline: {joke.punchline}")
    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_sync_streaming():
    """Demonstrate synchronous streaming with structured output."""
    print("\n=== Synchronous Streaming Example ===")

    chat_model = SecureGPTChat()

    # Create structured output model
    structured_llm = chat_model.with_structured_output(Story)

    messages = [HumanMessage(content="Write a short fantasy story about a magical library")]

    try:
        print("Streaming story generation...")
        print("Raw chunks:")

        # Note: When streaming with structured output, we get chunks first, then final parsed result
        collected_content = ""
        for chunk in structured_llm.stream(messages):
            if hasattr(chunk, "content"):
                # This is a regular streaming chunk
                collected_content += chunk.content
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, Story):
                # This is the final structured output
                print("\n\nStructured output received:")
                print(f"  Title: {chunk.title}")
                print(f"  Genre: {chunk.genre}")
                print(f"  Characters: {', '.join(chunk.characters)}")
                print(f"  Word count: {chunk.word_count}")
                print(f"  Plot: {chunk.plot[:100]}...")
                break

        # If we didn't get a structured output, show what we collected
        if collected_content and not isinstance(chunk, Story):
            print(f"\n\nCollected content ({len(collected_content)} characters):")
            print(f"Raw JSON: {collected_content[:200]}...")

    except Exception as e:
        print(f"Error in generation: {e}")


async def demonstrate_async_streaming():
    """Demonstrate asynchronous streaming with structured output."""
    print("\n=== Asynchronous Streaming Example ===")

    chat_model = SecureGPTChat()

    structured_llm = chat_model.with_structured_output(NewsArticle)

    messages = [HumanMessage(content="Write a news article about advances in AI technology")]

    try:
        print("Async streaming news article generation...")

        collected_content = ""
        async for chunk in structured_llm.astream(messages):
            if hasattr(chunk, "content"):
                # Regular streaming chunk
                collected_content += chunk.content
                print(f"  {chunk.content}", end="", flush=True)
            elif isinstance(chunk, NewsArticle):
                # Final structured output
                print("\n\nStructured article received:")
                print(f"  Headline: {chunk.headline}")
                print(f"  Category: {chunk.category}")
                print(f"  Urgency: {chunk.urgency}")
                print(f"  Summary: {chunk.summary}")
                print(f"  Key points: {len(chunk.key_points)} points")
                for i, point in enumerate(chunk.key_points, 1):
                    print(f"    {i}. {point}")

    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_streaming_with_error_handling():
    """Demonstrate streaming with proper error handling."""
    print("\n=== Streaming with Error Handling ===")

    chat_model = SecureGPTChat()

    # Use include_raw=True for better error handling
    structured_llm = chat_model.with_structured_output(Joke, include_raw=True)

    messages = [HumanMessage(content="Tell me a joke about streaming")]

    try:
        print("Streaming with error handling...")

        chunks = []
        final_result = None

        for chunk in structured_llm.stream(messages):
            if hasattr(chunk, "content"):
                chunks.append(chunk.content)
                print(f"  {chunk.content}", end="", flush=True)
            elif isinstance(chunk, dict):
                # This is the final result with raw and parsed
                final_result = chunk
                break

        print("\n\nProcessing final result...")

        if final_result:
            if final_result.get("parsing_error"):
                print(f"Parsing error: {final_result['parsing_error']}")
                print(f"Raw content: {final_result['raw'].content}")
            else:
                joke = final_result["parsed"]
                print("Successfully parsed joke:")
                print(f"  Setup: {joke.setup}")
                print(f"  Punchline: {joke.punchline}")
                print(f"  Rating: {joke.rating}")

    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_streaming_progress_tracking():
    """Demonstrate tracking progress during streaming."""
    print("\n=== Streaming Progress Tracking ===")

    chat_model = SecureGPTChat()

    structured_llm = chat_model.with_structured_output(Recipe)

    messages = [HumanMessage(content="Give me a detailed recipe for homemade pizza")]

    try:
        print("Streaming recipe with progress tracking...")

        import time

        start_time = time.time()
        chunk_count = 0
        total_chars = 0

        for chunk in structured_llm.stream(messages):
            if hasattr(chunk, "content"):
                chunk_count += 1
                total_chars += len(chunk.content)
                elapsed = time.time() - start_time

                # Show progress every 10 chunks
                if chunk_count % 10 == 0:
                    print(f"\n[Progress: {chunk_count} chunks, {total_chars} chars, {elapsed:.1f}s]")

                print(f"{chunk.content}", end="", flush=True)

            elif isinstance(chunk, Recipe):
                elapsed = time.time() - start_time
                print(f"\n\nStreaming completed in {elapsed:.1f}s")
                print(f"Total chunks: {chunk_count}, Total characters: {total_chars}")
                print(f"Final recipe: {chunk.name}")
                print(f"Ingredients: {len(chunk.ingredients)} items")
                print(f"Instructions: {len(chunk.instructions)} steps")

    except Exception as e:
        print(f"Error in generation: {e}")


def demonstrate_streaming_with_custom_handler():
    """Demonstrate custom handling during streaming."""
    print("\n=== Streaming with Custom Handler ===")

    chat_model = SecureGPTChat()

    structured_llm = chat_model.with_structured_output(PersonInfo)

    messages = [HumanMessage(content="Create a character profile for a detective story")]

    class StreamingHandler:
        def __init__(self):
            self.buffer = ""
            self.word_count = 0
            self.json_started = False

        def handle_chunk(self, chunk_content: str):
            self.buffer += chunk_content
            self.word_count += len(chunk_content.split())

            # Simple detection of JSON start
            if "{" in chunk_content and not self.json_started:
                self.json_started = True
                print("\n[JSON structure detected]")

            # Show word count every 20 words
            if self.word_count % 20 == 0:
                print(f"\n[Words: {self.word_count}]", end="")

        def get_stats(self):
            return {
                "total_chars": len(self.buffer),
                "word_count": self.word_count,
                "json_detected": self.json_started,
            }

    try:
        handler = StreamingHandler()

        print("Streaming with custom handler...")

        for chunk in structured_llm.stream(messages):
            if hasattr(chunk, "content"):
                handler.handle_chunk(chunk.content)
                print(f"{chunk.content}", end="", flush=True)

            elif isinstance(chunk, PersonInfo):
                stats = handler.get_stats()
                print("\n\nStreaming completed!")
                print(f"Statistics: {stats}")
                print(f"Character: {chunk.name}, Age: {chunk.age}")
                print(f"Occupation: {chunk.occupation}")
                print(f"Location: {chunk.location}")

    except Exception as e:
        print(f"Error in generation: {e}")


async def demonstrate_concurrent_streaming():
    """Demonstrate concurrent streaming of multiple requests."""
    print("\n=== Concurrent Streaming Example ===")

    import asyncio

    async def stream_joke():
        chat_model = SecureGPTChat()
        structured_llm = chat_model.with_structured_output(Joke)
        messages = [HumanMessage(content="Tell me a joke about programming")]

        print("Starting joke stream...")
        async for chunk in structured_llm.astream(messages):
            if isinstance(chunk, Joke):
                return f"Joke: {chunk.setup} - {chunk.punchline}"
        return "Joke stream completed"

    async def stream_recipe():
        chat_model = SecureGPTChat()
        structured_llm = chat_model.with_structured_output(Recipe)
        messages = [HumanMessage(content="Quick pasta recipe")]

        print("Starting recipe stream...")
        async for chunk in structured_llm.astream(messages):
            if isinstance(chunk, Recipe):
                return f"Recipe: {chunk.name} ({chunk.prep_time_minutes} min)"
        return "Recipe stream completed"

    try:
        # Run both streams concurrently
        results = await asyncio.gather(stream_joke(), stream_recipe(), return_exceptions=True)

        print("Concurrent streaming results:")
        for i, result in enumerate(results, 1):
            print(f"  Stream {i}: {result}")

    except Exception as e:
        print(f"Error in generation: {e}")


async def run_async_demos():
    """Run all async demonstrations in a single event loop."""
    await demonstrate_async_usage()
    await demonstrate_async_streaming()
    await demonstrate_concurrent_streaming()


def main():
    """Main function demonstrating various structured output scenarios."""
    print("SecureGPTChat Structured Output Demo")
    print("=" * 50)

    # Setup
    setup_environment()
    mock_auth_for_demo()

    # Run demonstrations
    demonstrate_joke_generation()
    demonstrate_recipe_generation()
    demonstrate_with_include_raw()
    demonstrate_error_handling()

    # New streaming demonstrations
    demonstrate_sync_streaming()
    demonstrate_streaming_with_error_handling()
    demonstrate_streaming_progress_tracking()
    demonstrate_streaming_with_custom_handler()

    print("\n=== Usage Tips ===")
    print("1. Always use BaseMessage objects (HumanMessage, SystemMessage, etc.)")
    print("2. Use include_raw=True for better error handling")
    print("3. Define clear Pydantic models with Field descriptions")
    print("4. Handle parsing errors gracefully in production code")
    print("5. Set appropriate temperature for your use case")
    print("6. Streaming provides real-time feedback but final parsing happens at the end")
    print("7. Use custom handlers for advanced streaming processing")
    print("8. Consider concurrent streaming for multiple requests")

    print("\n=== To run with real API ===")
    print("1. Set SECUREGPT_URL environment variable")
    print("2. Set up proper authentication (SECUREGPT_ACCESS_TOKEN or OneLogin)")
    print("3. Set SECUREGPT_MODEL_ID and SECUREGPT_PROVIDER")
    print("4. Run the script with real credentials")


if __name__ == "__main__":
    import asyncio

    # Run sync demos
    main()

    # Run async demos
    print("\nRunning async demos...")
    try:
        asyncio.run(run_async_demos())
    except Exception as e:
        print(f"Async demos completed (demo mode): {e}")
