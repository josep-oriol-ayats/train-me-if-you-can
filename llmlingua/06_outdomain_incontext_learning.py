# -*- coding: utf-8 -*-
"""
Out-of-Domain Testing: In-Context Learning (GSM8K)

Tests LLMLingua-2 on GSM8K math reasoning task (out-of-domain data).
This demonstrates the model's ability to compress few-shot examples while
preserving the reasoning patterns needed for math problem solving.

Target token count: 150 tokens
Uses force_reserve_digit to preserve numerical values in math problems.
"""

from config import llm_lingua, llm
from langchain_core.messages import HumanMessage
from datasets import load_dataset_builder, Split
import httpx
from pathlib import Path


print("=" * 80)
print("OUT-OF-DOMAIN TESTING: In-Context Learning (GSM8K)")
print("=" * 80)

# Download the prompt file with few-shot examples
print("\nDownloading GSM8K prompt examples...")
url = "https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt"
response = httpx.get(url)
response.raise_for_status()
Path("prompt_hardest.txt").write_text(response.text)
print("Downloaded successfully!")

# Load complex prompt with examples
prompt_complex = open("./prompt_hardest.txt").read()
print(f"Few-shot examples length: ~{len(prompt_complex.split())} words")

# Load GSM8K dataset using builder pattern
builder = load_dataset_builder("gsm8k", "main", trust_remote_code=True)
builder.download_and_prepare()
gsm8k_test = builder.as_dataset(split=Split.TEST)

# Select an example from GSM8K
question, answer = [gsm8k_test[2][key] for key in ["question", "answer"]]
print(f"\nTest Question: {question}")
print(f"Ground-truth Answer: {answer}")

# Test 1: Original prompt with SecureGPT
print("\n" + "-" * 80)
print("TEST 1: Original Prompt with Few-Shot Examples")
print("-" * 80)

instruction = "Please reference the following examples to answer the math question,\n"
prompt = instruction + prompt_complex + "\n\nQuestion: " + question

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=400, temperature=0, top_p=1, stop=["\n\n"])
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

# Test 2: Compressed prompt (target: 150 tokens)
print("\n" + "-" * 80)
print("TEST 2: Compressed Prompt (target_token=150, force_reserve_digit=True)")
print("-" * 80)

# Split examples for better compression
compressed_prompt = llm_lingua.compress_prompt(
    prompt_complex.split("\n\n"),
    target_token=150,
    force_tokens=["+", "-", "*", "×", "/", "÷", "=", "The answer is", "\n"],
    drop_consecutive=True,
    force_reserve_digit=True,
    use_context_level_filter=True,
)

print(
    f"Compressed examples length: ~{len(compressed_prompt['compressed_prompt'].split())} words"
)
print(f"Compression ratio: {compressed_prompt.get('ratio', 'N/A')}")

instruction = "Please reference the following examples to answer the math question,\n"
prompt = (
    instruction + compressed_prompt["compressed_prompt"] + "\n\nQuestion: " + question
)

message = [
    HumanMessage(content=prompt),
]

response = llm.invoke(message, max_tokens=400, temperature=0, top_p=1, stop=["\r\n"])
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
