"""Example of using SecureGPTChat with multimodal (image + text) inputs."""

import asyncio
import base64

from cai_securegpt_client.securegpt_chat import SecureGPTChat


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ask_about_image_with_dict(query: str, image_base64: str) -> str:
    """Ask a question about an image using dictionary message format.

    This demonstrates the exact format from your example.

    Args:
        query: The text question about the image
        image_base64: Base64 encoded image string

    Returns:
        The model's response
    """
    # Initialize the chat model
    chat_model = SecureGPTChat()

    # Create the multimodal message using dictionary format
    # This matches your example exactly
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "source_type": "base64",
                        "url": (
                            image_base64
                            if image_base64.startswith("data:")
                            else f"data:image/jpeg;base64,{image_base64}"
                        ),
                    },
                    "detail": "auto",
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        }
    ]

    # Send request to LLM - this now works with raw dictionary messages
    response = chat_model.invoke(messages)
    return response.content


def ask_about_image_with_helper(query: str, image_base64: str) -> str:
    """Ask a question about an image using the convenience helper method.

    Args:
        query: The text question about the image
        image_base64: Base64 encoded image string

    Returns:
        The model's response
    """
    # Initialize the chat model
    chat_model = SecureGPTChat()

    # Create the multimodal message using the helper method
    message = chat_model.create_multimodal_message(query, image_base64, detail="auto")

    # Send request to LLM
    response = chat_model.invoke([message])
    return response.content


async def ask_about_image_async(query: str, image_base64: str) -> str:
    """Async version of asking about an image.

    Args:
        query: The text question about the image
        image_base64: Base64 encoded image string

    Returns:
        The model's response
    """
    # Initialize the chat model
    chat_model = SecureGPTChat()

    # Create the multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "source_type": "base64",
                        "url": (
                            image_base64
                            if image_base64.startswith("data:")
                            else f"data:image/jpeg;base64,{image_base64}"
                        ),
                    },
                    "detail": "auto",
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        }
    ]

    # Send async request to LLM
    response = await chat_model.ainvoke(messages)
    return response.content


def stream_about_image(query: str, image_base64: str):
    """Stream response about an image.

    Args:
        query: The text question about the image
        image_base64: Base64 encoded image string

    Yields:
        Chunks of the response as they arrive
    """
    # Initialize the chat model
    chat_model = SecureGPTChat()

    # Create the multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "source_type": "base64",
                        "url": (
                            image_base64
                            if image_base64.startswith("data:")
                            else f"data:image/jpeg;base64,{image_base64}"
                        ),
                    },
                    "detail": "auto",
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        }
    ]

    # Stream the response
    for chunk in chat_model.stream(messages):
        yield chunk.content


async def stream_about_image_async(query: str, image_base64: str):
    """Async stream response about an image using helper method.

    Args:
        query: The text question about the image
        image_base64: Base64 encoded image string

    Yields:
        Chunks of the response as they arrive
    """
    # Initialize the chat model
    chat_model = SecureGPTChat()

    # Create the multimodal message using the helper method
    message = chat_model.create_multimodal_message(query, image_base64, detail="auto")

    # Stream the response asynchronously
    async for chunk in chat_model.astream([message]):
        yield chunk.content


def main():
    """Example usage of multimodal functionality."""
    # Example with a placeholder base64 image
    # In practice, you'd load this from a real image file
    sample_image_base64 = (
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAYEBQYFBAYGBQYHBwYIChA"
        "KCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC"
        "0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKC"
        "goKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAA"
        "EDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAA"
        "AAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQ"
        "EAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    )

    query = "What do you see in this image?"

    print("Testing multimodal functionality with SecureGPTChat...")
    print("=" * 50)

    # Test 1: Using dictionary format (matches your example)
    print("1. Using dictionary message format:")
    try:
        response1 = ask_about_image_with_dict(query, sample_image_base64)
        print(f"Response: {response1}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)

    # Test 2: Using helper method
    print("2. Using helper method:")
    try:
        response2 = ask_about_image_with_helper(query, sample_image_base64)
        print(f"Response: {response2}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)

    # Test 3: Streaming response
    print("3. Streaming response:")
    try:
        print("Streaming response:")
        for chunk in stream_about_image(query, sample_image_base64):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)

    # Test 4: Async streaming response with helper
    print("4. Async streaming response with helper:")

    async def test_async_stream():
        try:
            print("Async streaming response:")
            async for chunk in stream_about_image_async(query, sample_image_base64):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_async_stream())


if __name__ == "__main__":
    main()
