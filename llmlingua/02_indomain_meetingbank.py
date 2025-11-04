# -*- coding: utf-8 -*-
"""
In-Domain Testing: MeetingBank QA

Tests LLMLingua-2 on the MeetingBank dataset (in-domain data).
MeetingBank is the dataset used to train the compressor, so this demonstrates
performance on in-domain data with 33% compression rate (~2000 tokens).

The script compares:
1. Original prompt performance
2. Compressed prompt performance (33% compression rate)
"""

from config import llm_lingua, llm
from langchain_core.messages import HumanMessage
from datasets import load_dataset_builder, Split


print("=" * 80)
print("IN-DOMAIN TESTING: MeetingBank")
print("=" * 80)

# Load MeetingBank dataset using builder pattern
builder = load_dataset_builder("huuuyeah/meetingbank", trust_remote_code=True)
builder.download_and_prepare()
dataset = builder.as_dataset(split=Split.TEST)
context = dataset[0]["transcript"]

question = "What is the agenda item three resolution 31669 about?\nAnswer:"
reference = "Encouraging individualized tenant assessment."

print(f"\nReference answer: {reference}")
print(f"\nOriginal context length: ~{len(context.split())} words")

# Test 1: Original prompt with SecureGPT
print("\n" + "-" * 80)
print("TEST 1: Original Prompt")
print("-" * 80)

prompt = "\n\n".join([context, question])

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=100, temperature=0, top_p=1)
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

# Test 2: Compressed prompt (33% compression rate)
print("\n" + "-" * 80)
print("TEST 2: Compressed Prompt (rate=0.33)")
print("-" * 80)

compressed_prompt = llm_lingua.compress_prompt(
    context,
    rate=0.33,
    force_tokens=["!", ".", "?", "\n"],
    drop_consecutive=True,
)

print(
    f"Compressed context length: ~{len(compressed_prompt['compressed_prompt'].split())} words"
)
print(f"Compression ratio: {compressed_prompt.get('ratio', 'N/A')}")

prompt = "\n\n".join([compressed_prompt["compressed_prompt"], question])

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=100, temperature=0, top_p=1)
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
