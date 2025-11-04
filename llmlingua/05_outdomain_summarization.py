# -*- coding: utf-8 -*-
"""
Out-of-Domain Testing: Summarization (Government Reports)

Tests LLMLingua-2 on Government Report summarization from LongBench (out-of-domain data).
This demonstrates the model's ability to compress long government reports while
preserving the information needed for summarization tasks.

Target token count: 3000 tokens
"""

from config import llm_lingua, llm
from langchain_core.messages import HumanMessage
from datasets import load_dataset_builder, Split


# Dataset configuration
dataset2prompt = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
}

dataset2maxlen = {
    "gov_report": 512,
}

print("=" * 80)
print("OUT-OF-DOMAIN TESTING: Summarization (Government Reports)")
print("=" * 80)

# Load dataset using builder pattern
task = "gov_report"
builder = load_dataset_builder("THUDM/LongBench", task, trust_remote_code=True)
builder.download_and_prepare()
dataset = builder.as_dataset(split=Split.TEST)
sample = dataset[0]
context = sample["context"]
reference = sample["answers"]

print(
    f"\nReference summary: {reference[:200]}..."
    if len(reference) > 200
    else f"\nReference summary: {reference}"
)
print(f"Original context length: ~{len(context.split())} words")

# Test 1: Original prompt with SecureGPT
print("\n" + "-" * 80)
print("TEST 1: Original Prompt")
print("-" * 80)

prompt_format = dataset2prompt[task]
max_gen = int(dataset2maxlen[task])
prompt = prompt_format.format(**sample)

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=max_gen, temperature=0, top_p=1)
print(
    f"Response: {response.content[:300]}..."
    if len(response.content) > 300
    else f"Response: {response.content}"
)
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

# Test 2: Compressed prompt (target: 3000 tokens)
print("\n" + "-" * 80)
print("TEST 2: Compressed Prompt (target_token=3000)")
print("-" * 80)

compressed_prompt = llm_lingua.compress_prompt(
    context,
    target_token=3000,
    force_tokens=["!", ".", "?", "\n"],
    drop_consecutive=True,
)

print(
    f"Compressed context length: ~{len(compressed_prompt['compressed_prompt'].split())} words"
)
print(f"Compression ratio: {compressed_prompt.get('ratio', 'N/A')}")

prompt_format = dataset2prompt[task]
max_gen = int(dataset2maxlen[task])
sample["context"] = compressed_prompt["compressed_prompt"]
prompt = prompt_format.format(**sample)

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=max_gen, temperature=0, top_p=1)
print(
    f"Response: {response.content[:300]}..."
    if len(response.content) > 300
    else f"Response: {response.content}"
)
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
