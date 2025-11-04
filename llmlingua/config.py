# -*- coding: utf-8 -*-
"""
Shared Configuration for LLMLingua2 Examples

This module provides shared instances of the LLMLingua-2 prompt compressor
and SecureGPT client to be used across all example scripts.

LLMLingua-2 focuses on task-agnostic prompt compression for better generalizability
and efficiency. It is trained via data distillation from GPT-4 for token classification
with a BERT-level encoder, offering 3x-6x faster performance than LLMLingua.

Reference: https://aclanthology.org/2024.findings-acl.57/
Original Colab: https://colab.research.google.com/github/microsoft/LLMLingua/blob/main/examples/LLMLingua2.ipynb
"""

from llmlingua import PromptCompressor
from cai_securegpt_client import SecureGPTChat

# Initialize LLMLingua-2 with the MeetingBank model
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="mps",  # for mac use "mps", for GPU use "cuda", for CPU use "cpu"
)

# Initialize SecureGPT chat client
llm = SecureGPTChat()

