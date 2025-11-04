#!/usr/bin/env python3
"""Simple test to verify SecureGPT is working."""

from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage

# Create client
client = SecureGPTChat(
    model_id="gpt-4-turbo-2024-04-09",
    provider="openai",
    temperature=0.7
)

# Make a simple request
messages = [HumanMessage(content="What is 2+2?")]
response = client.invoke(messages)

print(f"Response type: {type(response)}")
print(f"Response: {response}")
print(f"Content: {response.content if hasattr(response, 'content') else 'No content'}")

