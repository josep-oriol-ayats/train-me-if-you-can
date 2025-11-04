"""
Direct DSPy LM wrapper for SecureGPTChat.

This module provides a clean DSPy-compatible LM interface that directly wraps
SecureGPTChat without going through LiteLLM.
"""

from typing import Any, Optional, List
from dspy.clients.lm import LM
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class SecureGPTLM(LM):
    """DSPy LM implementation that directly uses SecureGPTChat."""

    def __init__(
        self,
        model_id: str = "gpt-4-turbo-2024-04-09",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        secure_gpt_api_base: Optional[str] = None,
        secure_gpt_access_token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the SecureGPT LM.

        Args:
            model_id: The SecureGPT model ID
            provider: The provider (e.g., 'openai', 'mistral')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            secure_gpt_api_base: SecureGPT API base URL
            secure_gpt_access_token: Access token (if not provided, will use OneLogin)
            **kwargs: Additional parameters to pass to SecureGPTChat
        """
        # Initialize parent with the model string
        super().__init__(model=f"securegpt/{model_id}", temperature=temperature, max_tokens=max_tokens)

        # Initialize the SecureGPTChat client
        client_kwargs = {
            'model_id': model_id,
            'provider': provider,
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }

        if secure_gpt_api_base:
            client_kwargs['secure_gpt_api_base'] = secure_gpt_api_base
        if secure_gpt_access_token:
            client_kwargs['secure_gpt_access_token'] = secure_gpt_access_token

        self.client = SecureGPTChat(**client_kwargs)
        self.model_id = model_id
        self.provider = provider

    def _convert_messages(self, messages: List[dict]) -> List[Any]:
        """Convert DSPy message format to LangChain message objects."""
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        return langchain_messages

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[dict]] = None, **kwargs):
        """
        Make a completion request.

        Args:
            prompt: Simple string prompt (will be converted to messages)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            The response text or list of response texts
        """
        # Convert prompt to messages if needed
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]

        # Convert to LangChain format
        langchain_messages = self._convert_messages(messages)

        # Make the request
        response = self.client.invoke(langchain_messages)

        # The SecureGPTChat client returns an AIMessage object directly
        if hasattr(response, 'content'):
            return response.content

        # Fallback for ChatResult format
        if hasattr(response, 'generations') and response.generations:
            generation = response.generations[0]
            if hasattr(generation, 'message'):
                return generation.message.content
            elif hasattr(generation, 'text'):
                return generation.text

        # Last resort fallback
        return str(response)

    def basic_request(self, prompt: str, **kwargs):
        """Basic request interface for DSPy."""
        return self(prompt=prompt, **kwargs)

    async def __call_async__(self, prompt: Optional[str] = None, messages: Optional[List[dict]] = None, **kwargs):
        """
        Async completion request.

        Args:
            prompt: Simple string prompt (will be converted to messages)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            The response text
        """
        # Convert prompt to messages if needed
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]

        # Convert to LangChain format
        langchain_messages = self._convert_messages(messages)

        # Make the async request
        response = await self.client.ainvoke(langchain_messages)

        # The SecureGPTChat client returns an AIMessage object directly
        if hasattr(response, 'content'):
            return response.content

        # Fallback for ChatResult format
        if hasattr(response, 'generations') and response.generations:
            generation = response.generations[0]
            if hasattr(generation, 'message'):
                return generation.message.content
            elif hasattr(generation, 'text'):
                return generation.text

        # Last resort fallback
        return str(response)

