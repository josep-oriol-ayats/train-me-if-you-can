# -*- coding: utf-8 -*-
"""
LLMLingua2 Setup and Initialization

This script demonstrates the setup of the LLMLingua2 prompt compressor and SecureGPT client.
The actual instances are now imported from the shared config module.

LLMLingua-2 focuses on task-agnostic prompt compression for better generalizability
and efficiency. It is trained via data distillation from GPT-4 for token classification
with a BERT-level encoder, offering 3x-6x faster performance than LLMLingua.

Reference: https://aclanthology.org/2024.findings-acl.57/
Original Colab: https://colab.research.google.com/github/microsoft/LLMLingua/blob/main/examples/LLMLingua2.ipynb
"""

from config import llm_lingua, llm

print("LLMLingua-2 and SecureGPT initialized successfully!")
print("Model: microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
