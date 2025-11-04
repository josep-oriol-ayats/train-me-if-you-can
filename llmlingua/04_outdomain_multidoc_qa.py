# -*- coding: utf-8 -*-
"""
Out-of-Domain Testing: Multi-Document QA (TriviaQA)

Tests LLMLingua-2 on TriviaQA from LongBench (out-of-domain data).
This demonstrates the model's ability to compress and handle multiple passages
for question answering tasks.

Target token count: 2000 tokens
Uses context-level filtering for better multi-document handling.
"""

from config import llm_lingua, llm
from langchain_core.messages import HumanMessage
from datasets import load_dataset_builder, Split


# Dataset configuration
dataset2prompt = {
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
}

dataset2maxlen = {
    "triviaqa": 32,
}

print("=" * 80)
print("OUT-OF-DOMAIN TESTING: Multi-Document QA (TriviaQA)")
print("=" * 80)

# Load dataset using builder pattern
task = "triviaqa"
builder = load_dataset_builder("THUDM/LongBench", task, trust_remote_code=True)
builder.download_and_prepare()
dataset = builder.as_dataset(split=Split.TEST)
sample = dataset[0]
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

# Test 2: Compressed prompt (target: 2000 tokens)
print("\n" + "-" * 80)
print("TEST 2: Compressed Prompt (target_token=2000, context-level filter)")
print("-" * 80)

# Split context into passages for better compression
context_list = context.split("\nPassage:")
context_list = ["\nPassage:" + c for c in context_list]

print(f"Number of passages: {len(context_list)}")

compressed_prompt = llm_lingua.compress_prompt(
    context_list,
    target_token=2000,
    force_tokens=["\nPassage:", ".", "?", "\n"],
    drop_consecutive=True,
    use_context_level_filter=True,
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
