"""
DSPy configuration for SecureGPT integration.

This module provides configuration and setup utilities for using SecureGPT
with DSPy through the LiteLLM interface.
"""

import os
import dspy
from typing import Optional
import ssl
import warnings

# Disable SSL verification globally for corporate proxy/self-signed certificates
# This must be done before any SSL connections are made
_original_create_default_context = ssl.create_default_context

def _create_unverified_context(*args, **kwargs):
    """Create an SSL context that doesn't verify certificates."""
    context = _original_create_default_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# Monkey-patch SSL context creation
ssl.create_default_context = _create_unverified_context

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
warnings.filterwarnings('ignore', message='verify=False')


def configure_securegpt_lm(
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Configure and return a DSPy LM instance using SecureGPT.

    This function creates a direct DSPy-compatible wrapper around SecureGPTChat,
    bypassing LiteLLM for more reliable operation.

    Args:
        model_id: The SecureGPT model ID (e.g., 'gpt-4-turbo-2024-04-09').
                  Defaults to SECUREGPT_MODEL_ID env var or 'gpt-4-turbo-2024-04-09'.
        provider: The provider (e.g., 'openai', 'mistral').
                  Defaults to SECUREGPT_PROVIDER env var or 'openai'.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        api_base: SecureGPT API base URL. Defaults to SECUREGPT_URL env var.
        api_key: Access token. If not provided, will use SECUREGPT_ACCESS_TOKEN
                 env var or fall back to OneLogin authentication.
        **kwargs: Additional parameters to pass to SecureGPTChat.

    Returns:
        A configured LM instance ready to use with SecureGPT.

    Example:
        >>> import dspy
        >>> from dspy_securegpt.config import configure_securegpt_lm
        >>>
        >>> # Configure SecureGPT LM
        >>> lm = configure_securegpt_lm(
        ...     model_id="gpt-4-turbo-2024-04-09",
        ...     temperature=0.7
        ... )
        >>>
        >>> # Set as default LM for DSPy
        >>> dspy.configure(lm=lm)
        >>>
        >>> # Use with DSPy
        >>> response = lm("What is machine learning?")
    """
    # Import the direct wrapper
    try:
        from .securegpt_dspy_lm import SecureGPTLM
    except ImportError:
        from dspy_securegpt.securegpt_dspy_lm import SecureGPTLM

    # Get configuration from environment or parameters
    model_id = model_id or os.getenv("SECUREGPT_MODEL_ID", "gpt-4-turbo-2024-04-09")
    provider = provider or os.getenv("SECUREGPT_PROVIDER", "openai")
    api_base = api_base or os.getenv("SECUREGPT_URL")
    api_key = api_key or os.getenv("SECUREGPT_ACCESS_TOKEN")

    # Prepare kwargs for SecureGPTLM
    lm_kwargs = {
        "model_id": model_id,
        "provider": provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }

    # Add API configuration if provided
    if api_base:
        lm_kwargs["secure_gpt_api_base"] = api_base
    if api_key:
        lm_kwargs["secure_gpt_access_token"] = api_key

    # Create and return the LM instance
    return SecureGPTLM(**lm_kwargs)


def get_default_securegpt_lm():
    """
    Get a default SecureGPT LM instance using environment variables.

    This is a convenience function that creates an LM with all settings
    from environment variables.

    Environment Variables:
        SECUREGPT_MODEL_ID: Model ID (default: gpt-4-turbo-2024-04-09)
        SECUREGPT_PROVIDER: Provider (default: openai)
        SECUREGPT_URL: API base URL
        SECUREGPT_ACCESS_TOKEN: Access token (optional, falls back to OneLogin)
        SECUREGPT_TEMPERATURE: Temperature (default: 0.7)
        SECUREGPT_MAX_TOKENS: Max tokens (default: 4000)

    Returns:
        A configured LM instance.
    """
    return configure_securegpt_lm(
        temperature=float(os.getenv("SECUREGPT_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("SECUREGPT_MAX_TOKENS", "4000"))
    )

