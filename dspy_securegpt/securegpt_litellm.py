"""
LiteLLM-compatible wrapper for SecureGPTChat to enable DSPy integration.

This module provides a custom LiteLLM handler that wraps the SecureGPTChat client,
allowing it to be used seamlessly with DSPy's LM interface.
"""

from typing import Any, AsyncIterator, Iterator, Optional, Union
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ModelResponse, GenericStreamingChunk, Choices, Message, Usage
from cai_securegpt_client import SecureGPTChat
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


class SecureGPTLiteLLMHandler(litellm.CustomLLM):
    """Custom LiteLLM handler for SecureGPTChat integration with DSPy."""

    def __init__(self):
        super().__init__()
        self._client_cache = {}

    def _get_client(self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs) -> SecureGPTChat:
        """Get or create a cached SecureGPTChat client."""
        import os

        # Create a cache key based on configuration
        cache_key = (api_base or "", api_key or "")

        if cache_key not in self._client_cache:
            client_kwargs = {}
            if api_base:
                client_kwargs['secure_gpt_api_base'] = api_base
            if api_key:
                client_kwargs['secure_gpt_access_token'] = api_key

            # Extract additional parameters from kwargs or environment
            if 'temperature' in kwargs:
                client_kwargs['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                client_kwargs['max_tokens'] = kwargs['max_tokens']

            # Get model_id from kwargs or environment
            if 'model_id' in kwargs:
                client_kwargs['model_id'] = kwargs['model_id']
            elif os.getenv('SECUREGPT_MODEL_ID'):
                client_kwargs['model_id'] = os.getenv('SECUREGPT_MODEL_ID')

            # Get provider from kwargs or environment
            if 'provider' in kwargs:
                client_kwargs['provider'] = kwargs['provider']
            elif os.getenv('SECUREGPT_PROVIDER'):
                client_kwargs['provider'] = os.getenv('SECUREGPT_PROVIDER')

            self._client_cache[cache_key] = SecureGPTChat(**client_kwargs)

        return self._client_cache[cache_key]

    def _convert_messages(self, messages: list) -> list:
        """Convert LiteLLM message format to LangChain message objects."""
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
            elif role == "tool":
                langchain_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", "")
                ))

        return langchain_messages

    def _convert_to_model_response(self, response: Any, model: str) -> ModelResponse:
        """Convert LangChain ChatResult to LiteLLM ModelResponse format."""
        # Extract the response data
        if hasattr(response, 'generations') and response.generations:
            generation = response.generations[0]
            message = generation.message

            # Build the response in OpenAI format
            choices = [
                Choices(
                    finish_reason=generation.generation_info.get("finish_reason", "stop"),
                    index=0,
                    message=Message(
                        content=message.content,
                        role="assistant",
                        tool_calls=getattr(message, "tool_calls", None)
                    )
                )
            ]

            # Extract usage information
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            usage = Usage(
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0)
            )

            model_response = ModelResponse(
                id=f"chatcmpl-{hash(message.content)}",
                choices=choices,
                created=int(litellm.utils.get_utc_datetime().timestamp()),
                model=model,
                object="chat.completion",
                usage=usage
            )

            return model_response

        # Fallback for unexpected response format
        return ModelResponse(
            id="chatcmpl-unknown",
            choices=[],
            created=int(litellm.utils.get_utc_datetime().timestamp()),
            model=model,
            object="chat.completion"
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[float] = None,
        client: Optional[HTTPHandler] = None
    ) -> Union[ModelResponse, Iterator[GenericStreamingChunk]]:
        """Synchronous completion method for LiteLLM."""
        # Extract model_id from model string (format: "securegpt/model-id")
        model_parts = model.split("/")
        model_id = model_parts[-1] if len(model_parts) > 1 else model

        # Get or create client
        client_kwargs = {
            'model_id': model_id,
            **optional_params
        }
        securegpt_client = self._get_client(api_key, api_base, **client_kwargs)

        # Convert messages
        langchain_messages = self._convert_messages(messages)

        # Check if streaming is requested
        if optional_params.get("stream", False):
            return self.streaming(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client
            )

        # Make the call
        response = securegpt_client.invoke(langchain_messages)

        # Convert to LiteLLM format
        return self._convert_to_model_response(response, model)

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[float] = None,
        client: Optional[AsyncHTTPHandler] = None
    ) -> Union[ModelResponse, AsyncIterator[GenericStreamingChunk]]:
        """Asynchronous completion method for LiteLLM."""
        # Extract model_id from model string
        model_parts = model.split("/")
        model_id = model_parts[-1] if len(model_parts) > 1 else model

        # Get or create client
        client_kwargs = {
            'model_id': model_id,
            **optional_params
        }
        securegpt_client = self._get_client(api_key, api_base, **client_kwargs)

        # Convert messages
        langchain_messages = self._convert_messages(messages)

        # Check if streaming is requested
        if optional_params.get("stream", False):
            # For streaming, return an async generator - don't await it
            return self.astreaming(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client
            )

        # Make the async call
        response = await securegpt_client.ainvoke(langchain_messages)

        # Convert to LiteLLM format
        return self._convert_to_model_response(response, model)

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[float] = None,
        client: Optional[HTTPHandler] = None
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming method for LiteLLM."""
        # Extract model_id from model string
        model_parts = model.split("/")
        model_id = model_parts[-1] if len(model_parts) > 1 else model

        # Get or create client
        client_kwargs = {
            'model_id': model_id,
            **optional_params
        }
        securegpt_client = self._get_client(api_key, api_base, **client_kwargs)

        # Convert messages
        langchain_messages = self._convert_messages(messages)

        # Stream the response
        for chunk in securegpt_client.stream(langchain_messages):
            # ChatGenerationChunk has a message attribute
            delta_content = getattr(chunk.message, 'content', '') if hasattr(chunk, 'message') else ""
            generation_info = getattr(chunk, 'generation_info', {}) or {}

            yield GenericStreamingChunk(
                text=delta_content,
                tool_use=None,
                is_finished=generation_info.get("finish_reason") is not None,
                finish_reason=generation_info.get("finish_reason"),
                usage=None,
                index=0
            )

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout: Optional[float] = None,
        client: Optional[AsyncHTTPHandler] = None
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Asynchronous streaming method for LiteLLM."""
        # Extract model_id from model string
        model_parts = model.split("/")
        model_id = model_parts[-1] if len(model_parts) > 1 else model

        # Get or create client
        client_kwargs = {
            'model_id': model_id,
            **optional_params
        }
        securegpt_client = self._get_client(api_key, api_base, **client_kwargs)

        # Convert messages
        langchain_messages = self._convert_messages(messages)

        # Stream the response asynchronously
        async for chunk in securegpt_client.astream(langchain_messages):
            # ChatGenerationChunk has a message attribute
            delta_content = getattr(chunk.message, 'content', '') if hasattr(chunk, 'message') else ""
            generation_info = getattr(chunk, 'generation_info', {}) or {}

            yield GenericStreamingChunk(
                text=delta_content,
                tool_use=None,
                is_finished=generation_info.get("finish_reason") is not None,
                finish_reason=generation_info.get("finish_reason"),
                usage=None,
                index=0
            )


# Register the handler with LiteLLM
def register_securegpt_with_litellm():
    """Register SecureGPT as a custom provider in LiteLLM."""
    litellm.custom_provider_map = [
        {"provider": "securegpt", "custom_handler": SecureGPTLiteLLMHandler()}
    ] + (litellm.custom_provider_map or [])


# Auto-register on import
register_securegpt_with_litellm()

