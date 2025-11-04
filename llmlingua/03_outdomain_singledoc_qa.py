# -*- coding: utf-8 -*-
"""
Out-of-Domain Testing: Single-Document QA (NarrativeQA)

Tests LLMLingua-2 on NarrativeQA from LongBench (out-of-domain data).
This demonstrates the model's generalization ability on single-document QA tasks
where the context is a story or movie script.

Target token count: 3000 tokens
"""

from config import llm_lingua, llm
from langchain_core.messages import HumanMessage
from datasets import load_dataset_builder, Split


# Dataset configuration
dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
}

dataset2maxlen = {
    "narrativeqa": 128,
}

print("=" * 80)
print("OUT-OF-DOMAIN TESTING: Single-Document QA (NarrativeQA)")
print("=" * 80)

# Load dataset using builder pattern
task = "narrativeqa"
builder = load_dataset_builder("THUDM/LongBench", task, trust_remote_code=True)
builder.download_and_prepare()
dataset = builder.as_dataset(split=Split.TEST)
sample = dataset[3]
context = sample["context"]
reference = sample["answers"]

print(f"\nReference answers: {reference}")
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
print(f"Response: {response.content}")
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
print(f"Response: {response.content}")
print(f"Token usage: {response.response_metadata.get('token_usage', {})}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
