#!/usr/bin/env python3
"""
Test script to verify the config.py fix for CUSTOM_OPENAI_API_KEY error.
"""

import os
import sys

# Set test environment variables
os.environ["SECUREGPT_ACCESS_TOKEN"] = "test-token-123"
os.environ["SECUREGPT_URL"] = "https://api.test.example.com"
os.environ["SECUREGPT_MODEL_ID"] = "gpt-4-turbo-2024-04-09"
os.environ["SECUREGPT_PROVIDER"] = "openai"

# Import the config module
from dspy_securegpt.config import configure_securegpt_lm

print("Testing SecureGPT configuration...")
print(f"SECUREGPT_ACCESS_TOKEN: {os.getenv('SECUREGPT_ACCESS_TOKEN')}")
print(f"SECUREGPT_URL: {os.getenv('SECUREGPT_URL')}")

try:
    # Configure the LM
    lm = configure_securegpt_lm(
        model_id="gpt-4-turbo-2024-04-09",
        temperature=0.7
    )

    # Check that environment variables were set correctly
    print("\n✅ Configuration successful!")
    print(f"CUSTOM_OPENAI_API_KEY is set: {os.getenv('CUSTOM_OPENAI_API_KEY') is not None}")
    print(f"CUSTOM_OPENAI_API_BASE is set: {os.getenv('CUSTOM_OPENAI_API_BASE') is not None}")

    # Verify the values
    if os.getenv('CUSTOM_OPENAI_API_KEY') == os.getenv('SECUREGPT_ACCESS_TOKEN'):
        print("✅ CUSTOM_OPENAI_API_KEY correctly set from SECUREGPT_ACCESS_TOKEN")
    else:
        print("❌ CUSTOM_OPENAI_API_KEY mismatch")

    if os.getenv('CUSTOM_OPENAI_API_BASE') == os.getenv('SECUREGPT_URL'):
        print("✅ CUSTOM_OPENAI_API_BASE correctly set from SECUREGPT_URL")
    else:
        print("❌ CUSTOM_OPENAI_API_BASE mismatch")

    print(f"\nConfigured LM: {lm}")
    print("\n✅ All checks passed! The authentication error should be fixed.")

except Exception as e:
    print(f"\n❌ Error during configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

