#!/usr/bin/env python3
"""Test SecureGPTChat directly."""

from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

print("Creating SecureGPTChat client...")
client = SecureGPTChat(
    model_id="gpt-4-turbo-2024-04-09",
    provider="openai",
    temperature=0.7,
    max_tokens=100
)

print("Making a request...")
messages = [HumanMessage(content="What is 2+2? Answer in one word.")]

try:
    response = client.invoke(messages)
    print(f"Response type: {type(response)}")
    print(f"Response: {response}")
    if hasattr(response, 'content'):
        print(f"Content: {response.content}")
        print(f"Content type: {type(response.content)}")
        print(f"Content length: {len(response.content)}")
    else:
        print("No content attribute")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")

