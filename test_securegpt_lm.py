#!/usr/bin/env python3
"""Test SecureGPTLM basic functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dspy_securegpt.config import configure_securegpt_lm
import dspy

print("Configuring SecureGPT LM...")
lm = configure_securegpt_lm(
    model_id="gpt-4-turbo-2024-04-09",
    temperature=0.7,
    max_tokens=100
)

print("Configuring DSPy...")
dspy.configure(lm=lm)

print("Making a simple call...")
try:
    result = lm("What is 2+2? Answer in one word.")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"Result repr: {repr(result)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTrying with messages...")
try:
    result2 = lm(messages=[{"role": "user", "content": "What is 3+3? Answer in one word."}])
    print(f"Result2 type: {type(result2)}")
    print(f"Result2: {result2}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTrying DSPy Predict...")
try:
    class SimpleSig(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    predictor = dspy.Predict(SimpleSig)
    response = predictor(question="What is 5+5?")
    print(f"Response: {response}")
    print(f"Answer: {response.answer}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")

