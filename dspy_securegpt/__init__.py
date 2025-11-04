"""
DSPy integration for SecureGPT.

This package provides utilities to use SecureGPT models with DSPy through LiteLLM.
"""

from .config import configure_securegpt_lm, get_default_securegpt_lm
from . import securegpt_litellm

__all__ = [
    "configure_securegpt_lm",
    "get_default_securegpt_lm",
    "securegpt_litellm",
]

