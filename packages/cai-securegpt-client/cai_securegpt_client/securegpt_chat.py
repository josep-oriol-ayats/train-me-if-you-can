"""Custom Chat Model from langchain BaseChatModel compatible with SecureGPT API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import (
    Any,
    AsyncContextManager,
    Dict,
    List,
    Literal,
    Type,
    Union,
    cast,
)

import httpx
from httpx import AsyncClient, Client
from httpx_sse import EventSource, aconnect_sse, connect_sse
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from .onelogin_auth import OneloginAuth  # type: ignore


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Transforms a message delta to a messagen chunk
    casted to its corresponding type.

    Arguments:
        _dict -- The delta dictionary.
        default_class -- The default class for the delta.

    Returns:
        A message chunk.
    """
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")  # Handle None content by using empty string
    tool_calls = _dict.get("tool_calls")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        chunk = AIMessageChunk(content=content)
        if tool_calls:
            chunk.tool_calls = tool_calls
        return chunk
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message cating it to
    its corresponding type depending on the role.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    content = _dict.get("content") or ""  # Handle None content by using empty string
    tool_calls = _dict.get("tool_calls")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        message = AIMessage(content=content)
        if tool_calls:
            # Convert tool calls to LangChain format
            converted_tool_calls = []
            for tool_call in tool_calls:
                # Parse arguments from JSON string to dict
                args_str = tool_call["function"]["arguments"]
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}

                # Create ToolCall using the proper constructor
                from langchain_core.messages import ToolCall as ToolCallType

                tool_call_obj = ToolCallType(
                    name=tool_call["function"]["name"],
                    args=args,
                    id=tool_call["id"],
                )
                converted_tool_calls.append(tool_call_obj)
            message.tool_calls = converted_tool_calls
        return message
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, tool_call_id=_dict.get("tool_call_id"))
    else:
        return ChatMessage(content=content, role=role or "user")


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: dict[str, Any]

    # Handle tool message
    if isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    # First check if it's a valid message type
    elif isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                # ToolCall is a TypedDict, so access as dictionary
                args = tool_call["args"]
                args_str: str
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                elif isinstance(args, str):
                    args_str = args
                else:
                    args_str = "{}"

                tool_calls.append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": args_str,
                        },
                    }
                )
            message_dict["tool_calls"] = tool_calls
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    # Handle multimodal content (list of content items) after we know it's a valid message
    if hasattr(message, "content") and isinstance(message.content, list):
        message_dict["content"] = message.content

    return message_dict


def _create_retry_decorator(
    llm: SecureGPTChat,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions."""
    errors = [httpx.RequestError, httpx.StreamError]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries, run_manager=run_manager)


async def _aiter_sse(
    event_source_mgr: AsyncContextManager[EventSource],
) -> AsyncIterator[dict]:
    """Iterate over the server-sent events."""
    async with event_source_mgr as event_source:
        async for event in event_source.aiter_sse():
            if event.data == "[DONE]":
                return
            if event.data == "":
                continue
            yield event.json()


async def acompletion_with_retry(
    llm: SecureGPTChat,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call with enhanced retry logic."""
    import asyncio

    last_exception = None

    for attempt in range(llm.max_retries + 1):
        try:
            headers = {}
            if llm.auth_manager.get_access_token_async is not None:
                access_token = await llm.auth_manager.get_access_token_async()
                headers["Authorization"] = f"Bearer {access_token}"
            else:
                headers["Authorization"] = f"Bearer {llm.secure_gpt_access_token}"

            url = generate_url(
                base_url=llm.secure_gpt_api_base,
                provider=llm.provider,
                model_id=llm.model_id,
                openai_api_ver=llm.openai_apiver,
            )

            if kwargs.get("stream"):
                event_source = aconnect_sse(
                    llm.async_client,
                    "POST",
                    url=url,
                    json=kwargs,
                    headers=headers,
                )
                return _aiter_sse(event_source)
            else:
                response = await llm.async_client.post(
                    url=url,
                    json=kwargs,
                    headers=headers,
                )

                # Check if we should retry based on status code
                if response.status_code in llm.retry_on_status_codes and attempt < llm.max_retries:
                    delay = llm._exponential_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {response.text}",
                        request=response.request,
                        response=response,
                    )

                return response.json()

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_exception = e
            if attempt < llm.max_retries:
                delay = llm._exponential_backoff(attempt)
                await asyncio.sleep(delay)
                continue
            else:
                # On final attempt, raise the exception
                if hasattr(e, "response") and e.response is not None:
                    raise ValueError(f"Error invoking secureGPT completions endpoint: {e}") from e
                else:
                    raise ValueError(f"Network error calling secureGPT: {e}") from e
        except Exception as e:
            # For non-retryable exceptions, raise immediately
            raise ValueError(f"Unexpected error calling secureGPT: {e}") from e

    # This shouldn't be reached, but just in case
    if last_exception:
        raise ValueError(f"Max retries exceeded: {last_exception}")
    else:
        raise ValueError("Unexpected error: no exception recorded")


def generate_url(base_url: str, provider: str, model_id: str, openai_api_ver: str = "2024-10-21") -> str:
    """Generates the url to call the secure gpt chat completions API.

    Arguments:
        base_url -- The secure gpt hub base url
        provider -- The model provider (openai, mistral...)
        model_id -- The model id (https://pages.github.axa.com/axa-go-getd/secure-gpt-model-hub/offerings/models/)

    Keyword Arguments:
        openai_api_ver -- openai API version (default: {"2024-10-21"})

    Returns:
        The url
    """
    api_version_param = f"?api-version={openai_api_ver}" if provider == "openai" else ""
    url = f"{base_url}/providers/{provider}/deployments/{model_id}/chat/completions"
    url = f"{url}{api_version_param}"
    return url


class SecureGPTChat(BaseChatOpenAI):
    """A custom chat model class that uses the Secure GPT API.

    deployment_id describes the name of the model in Secure GPT API documentation.

    """

    _BASE_HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    auth_manager: OneloginAuth = OneloginAuth()
    secure_gpt_access_token: str | None = None
    secure_gpt_api_base: str = os.getenv(
        "SECUREGPT_URL",
        "https://api.se.axa-go.applications.services.axa-tech.intraxa/ago-m365-securegpt-hub-v1-vrs",
    )
    model_id: str = os.getenv("SECUREGPT_MODEL_ID", "gpt-4-turbo-2024-04-09")
    provider: str = os.getenv("SECUREGPT_PROVIDER", "openai")
    openai_apiver: str = os.getenv("SECUREGPT_OPENAI_API_VERSION", "2024-10-21")
    max_retries: int = int(os.getenv("SECUREGPT_MAX_RETRIES", "2"))
    timeout: int = int(os.getenv("SECUREGPT_TIMEOUT", "120"))
    temperature: float = float(os.getenv("SECUREGPT_TEMPERATURE", "0.7"))

    # Add retry configuration fields
    retry_on_status_codes: List[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_backoff_factor: float = Field(default=2.0)
    retry_jitter: bool = Field(default=True)

    # Add optional attributes for tools and structured output
    _response_format: dict[str, Any] | None = None
    _tools: list[dict[str, Any]] | None = None
    _tool_choice: str | dict[str, Any] | None = None

    client: Client = httpx.Client(
        base_url=secure_gpt_api_base,
        headers=_BASE_HEADERS,
        timeout=timeout,
        verify=False,
    )
    async_client: AsyncClient = httpx.AsyncClient(
        base_url=secure_gpt_api_base,
        headers=_BASE_HEADERS,
        timeout=timeout,
        verify=False,
    )

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling the API."""
        defaults: dict[str, Any] = {}
        return defaults

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    def _llm_type(self):
        """Return a string identifier of the model."""
        return "SecureGPT-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "secure_gpt"]

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"  # Changed from "forbid" to "allow" to support dynamic attributes for testing
        arbitrary_types_allowed = True

    def handle_events(self, event_source):
        """Handle event_source."""
        for event in event_source.iter_sse():
            if event.data == "":
                continue
            if event.data == "[DONE]":
                return
            yield event.json()

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = self.retry_backoff_factor**attempt
        if self.retry_jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 50% jitter
        return min(delay, 60)  # Cap at 60 seconds

    def _extract_retry_after_seconds(self, error_text: str) -> float | None:
        """Extract retry-after seconds from error message.

        Parses error messages like:
        "Please retry after 43 seconds."

        Args:
            error_text: The error message text

        Returns:
            Number of seconds to wait, or None if not found
        """
        import re

        # Pattern to match "retry after X seconds" or "Please retry after X seconds"
        pattern = r"retry after (\d+) seconds?"
        match = re.search(pattern, error_text, re.IGNORECASE)

        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return None

        return None

    def _is_429_error(self, error_message: str) -> bool:
        """Check if error message contains HTTP 429 status."""
        return "HTTP 429" in error_message or "429" in error_message

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call with enhanced retry logic."""

        def _completion_with_retry(**kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(self.max_retries + 1):
                try:
                    if self.auth_manager.get_access_token is not None:
                        headers = {"Authorization": f"Bearer {self.auth_manager.get_access_token()}"}
                    else:
                        headers = {"Authorization": f"Bearer {self.secure_gpt_access_token}"}

                    url = generate_url(
                        base_url=self.secure_gpt_api_base,
                        provider=self.provider,
                        model_id=self.model_id,
                        openai_api_ver=self.openai_apiver,
                    )

                    if kwargs.get("stream"):

                        def iter_sse(url=url, headers=headers) -> Iterator[dict]:
                            with connect_sse(
                                self.client,
                                "POST",
                                url=url,
                                headers=headers,
                                json=kwargs,
                            ) as event_source:
                                yield from self.handle_events(event_source)

                        return iter_sse()
                    else:
                        response = self.client.post(
                            url=url,
                            headers=headers,
                            json=kwargs,
                        )

                        # Check if we should retry based on status code
                        if response.status_code in self.retry_on_status_codes and attempt < self.max_retries:
                            delay = self._exponential_backoff(attempt)
                            time.sleep(delay)
                            continue

                        if response.status_code >= 400:
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {response.text}",
                                request=response.request,
                                response=response,
                            )

                        return response.json()

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        delay = self._exponential_backoff(attempt)
                        time.sleep(delay)
                        continue
                    else:
                        # On final attempt, raise the exception
                        if hasattr(e, "response") and e.response is not None:
                            raise ValueError(f"Error invoking secureGPT completions endpoint: {e}") from e
                        else:
                            raise ValueError(f"Network error calling secureGPT: {e}") from e
                except Exception as e:
                    # For non-retryable exceptions, raise immediately
                    raise ValueError(f"Unexpected error calling secureGPT: {e}") from e

            # This shouldn't be reached, but just in case
            if last_exception:
                raise ValueError(f"Max retries exceeded: {last_exception}")
            else:
                raise ValueError("Unexpected error: no exception recorded")

        return _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop=stop)
        params = {**params, **kwargs}

        # Wrap with 429 retry logic
        for retry_attempt in range(self.max_retries + 1):
            try:
                response = self.completion_with_retry(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except ValueError as e:
                error_msg = str(e)
                # Check if this is a 429 error and we have retries left
                if self._is_429_error(error_msg) and retry_attempt < self.max_retries:
                    buffer = 2  # Add a small buffer to the wait time
                    retry_after = self._extract_retry_after_seconds(error_msg) + buffer
                    if retry_after:
                        print(f"Rate limit reached. Waiting {retry_after} seconds before retry...")
                        time.sleep(retry_after)
                        continue
                # Re-raise if not 429, no retries left, or no delay found
                raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio

        message_dicts, params = self._create_message_dicts(messages, stop=stop)
        params = {**params, **kwargs}

        # Wrap with 429 retry logic
        for retry_attempt in range(self.max_retries + 1):
            try:
                response = await acompletion_with_retry(self, messages=message_dicts, **params)
                return self._create_chat_result(response)
            except ValueError as e:
                error_msg = str(e)
                # Check if this is a 429 error and we have retries left
                if self._is_429_error(error_msg) and retry_attempt < self.max_retries:
                    retry_after = self._extract_retry_after_seconds(error_msg)
                    if retry_after:
                        print(f"Rate limit reached. Waiting {retry_after} seconds before retry...")
                        await asyncio.sleep(retry_after)
                        continue
                # Re-raise if not 429, no retries left, or no delay found
                raise

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop=stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.completion_with_retry(messages=message_dicts, **params):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            delta = choice["delta"]
            new_chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = new_chunk.__class__
            finish_reason = choice.get("finish_reason")
            gen_chunk = ChatGenerationChunk(
                message=new_chunk,
                generation_info=({"finish_reason": finish_reason} if finish_reason is not None else None),
            )
            if run_manager:
                run_manager.on_llm_new_token(token=cast(str, new_chunk.content), chunk=gen_chunk)
            yield gen_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop=stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async_iter = await acompletion_with_retry(self, messages=message_dicts, run_manager=run_manager, **params)
        async for chunk in async_iter:
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            delta = choice["delta"]
            new_chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = new_chunk.__class__
            finish_reason = choice.get("finish_reason")
            gen_chunk = ChatGenerationChunk(
                message=new_chunk,
                generation_info=({"finish_reason": finish_reason} if finish_reason is not None else None),
            )
            if run_manager:
                await run_manager.on_llm_new_token(token=cast(str, new_chunk.content), chunk=gen_chunk)
            yield gen_chunk

    def __str__(self):
        """Provides a string representation of the ChatSecureGPT instance,
        showing the deployment ID.

        Returns:
            str: The string representation of the instance.
        """
        return f"ChatSecureGPT(deployment_id={self.model_id})"

    def _create_chat_result(
        self, response: dict[Any, Any] | BaseModel, generation_info: dict[str, Any] | None = None
    ) -> ChatResult:
        generations = []
        # Handle both dict and BaseModel responses
        if isinstance(response, BaseModel):
            response_dict = response.model_dump()
        else:
            response_dict = response

        for res in response_dict["choices"]:
            finish_reason = res.get("finish_reason")
            gen = ChatGeneration(
                message=_convert_dict_to_message(res["message"]),
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)
        token_usage = response_dict.get("usage", {})

        llm_output = {"token_usage": token_usage, "model": self.model_id}
        return ChatResult(generations=generations, llm_output=llm_output)

    def with_structured_output(
        self,
        schema: Union[dict[str, Any], type[BaseModel], type, None] = None,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Union["SecureGPTChat", Any]:
        """Model configured to return structured output.

        Args:
            schema: The output schema as a dict or a Pydantic model.
            include_raw: Whether to include the raw response in the output.
            **kwargs: Additional arguments.

        Returns:
            A configured model or runnable that returns structured output.
        """
        if kwargs:
            raise ValueError(f"Received unsupported arguments: {kwargs}")

        # Handle Pydantic models
        if schema is not None and isinstance(schema, type) and is_basemodel_subclass(schema):
            # Remove the unused assignment
            pass
        else:
            # Remove the unused assignment
            pass

        # Create a new instance with response format configured
        # Use model_copy without deep=True to avoid copying non-serializable objects
        llm = self.model_copy()
        llm._response_format = {"type": "json_object"}

        if include_raw:
            return _StructuredOutputParser(
                llm=llm,
                schema=schema,
                include_raw=True,
            )
        else:
            return _StructuredOutputParser(
                llm=llm,
                schema=schema,
                include_raw=False,
            )

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop

        # Add response format if configured
        if hasattr(self, "_response_format") and self._response_format:
            params["response_format"] = self._response_format

        # Add tools if configured
        if hasattr(self, "_tools") and self._tools:
            params["tools"] = self._tools

        # Add tool choice if configured
        if hasattr(self, "_tool_choice") and self._tool_choice:
            params["tool_choice"] = self._tool_choice

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Literal["auto", "none", "required", "any"] | dict[Any, Any] | str | bool | None = "auto",
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> "SecureGPTChat":
        """Bind tools to the model.

        Args:
            tools: A list of tools to bind to the model. Can be:
                - Dictionary with tool schema
                - Pydantic model class
                - Python function
                - BaseTool instance
            tool_choice: How the model should choose tools. Can be:
                - "auto": Let the model decide
                - "none": Don't use tools
                - {"type": "function", "function": {"name": "tool_name"}}: Force specific tool
            **kwargs: Additional arguments

        Returns:
            A new SecureGPTChat instance with tools bound
        """
        formatted_tools = []

        for tool in tools:
            if isinstance(tool, dict):
                # Already formatted tool schema
                formatted_tools.append(tool)
            elif isinstance(tool, BaseTool):
                # LangChain BaseTool
                formatted_tools.append(convert_to_openai_tool(tool))
            elif isinstance(tool, type) and is_basemodel_subclass(tool):
                # Pydantic model
                formatted_tools.append(convert_to_openai_tool(tool))
            elif callable(tool):
                # Python function
                formatted_tools.append(convert_to_openai_tool(tool))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        # Create a new instance with tools bound
        llm = self.model_copy()
        llm._tools = formatted_tools

        # Convert tool_choice to the expected type
        if isinstance(tool_choice, (str, dict)):
            llm._tool_choice = tool_choice
        elif tool_choice is None:
            llm._tool_choice = None
        else:
            # Convert other types to string
            llm._tool_choice = str(tool_choice)

        return llm

    def with_tools(
        self,
        tools: list[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        **kwargs: Any,
    ) -> "SecureGPTChat":
        """Alias for bind_tools for compatibility.

        Args:
            tools: A list of tools to bind to the model
            tool_choice: How the model should choose tools
            **kwargs: Additional arguments

        Returns:
            A new SecureGPTChat instance with tools bound
        """
        return self.bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def _preprocess_messages(self, input: Any) -> List[BaseMessage]:
        """Preprocess input to ensure it's a list of BaseMessage objects.

        This method handles the conversion of raw dictionary messages to BaseMessage objects
        while maintaining compatibility with LangChain's interface expectations.

        Args:
            input: Can be a list of BaseMessage objects or raw message dictionaries

        Returns:
            List of BaseMessage objects
        """
        # Handle raw message dictionaries (for multimodal support)
        if isinstance(input, list) and len(input) > 0 and isinstance(input[0], dict):
            # Convert dictionary messages to BaseMessage objects
            messages: List[BaseMessage] = []
            for msg_dict in input:
                if msg_dict.get("role") == "user":
                    content = msg_dict.get("content", "")
                    # Handle multimodal content properly
                    if isinstance(content, list):
                        # Ensure we have the correct type for multimodal content
                        messages.append(HumanMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=str(content)))
                elif msg_dict.get("role") == "assistant":
                    ai_message = AIMessage(content=msg_dict.get("content", ""))
                    # Handle tool calls in dictionary format
                    if "tool_calls" in msg_dict:
                        converted_tool_calls = []
                        for tool_call in msg_dict["tool_calls"]:
                            converted_tool_calls.append(
                                ToolCall(
                                    name=tool_call["function"]["name"],
                                    args=tool_call["function"]["arguments"],
                                    id=tool_call["id"],
                                )
                            )
                        ai_message.tool_calls = converted_tool_calls
                    messages.append(ai_message)
                elif msg_dict.get("role") == "system":
                    messages.append(SystemMessage(content=msg_dict.get("content", "")))
                elif msg_dict.get("role") == "tool":
                    messages.append(
                        ToolMessage(
                            content=msg_dict.get("content", ""),
                            tool_call_id=msg_dict.get("tool_call_id"),
                        )
                    )
                else:
                    messages.append(
                        ChatMessage(
                            content=msg_dict.get("content", ""),
                            role=msg_dict.get("role", "user"),
                        )
                    )
            return messages
        elif isinstance(input, list):
            # Already BaseMessage objects, return as-is
            return input
        else:
            # Handle other input types by converting to BaseMessage list
            if hasattr(input, "to_messages"):
                return input.to_messages()
            elif isinstance(input, str):
                return [HumanMessage(content=input)]
            else:
                # Try to convert sequence to list
                try:
                    return list(input)
                except (TypeError, ValueError):
                    raise ValueError(f"Expected list of BaseMessage or dict objects, got {type(input)}") from None

    def invoke(self, input, config=None, **kwargs):
        """Invoke the model with input that can be messages or raw message dictionaries.

        Args:
            input: Can be a list of BaseMessage objects or raw message dictionaries
            config: Configuration for the invocation
            **kwargs: Additional arguments

        Returns:
            The model response
        """
        # Preprocess input to ensure compatibility
        processed_input = self._preprocess_messages(input)

        # Call the parent invoke method with processed input
        return super().invoke(processed_input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        """Async invoke the model with input that can be messages or raw message dictionaries.

        Args:
            input: Can be a list of BaseMessage objects or raw message dictionaries
            config: Configuration for the invocation
            **kwargs: Additional arguments

        Returns:
            The model response
        """
        # Preprocess input to ensure compatibility
        processed_input = self._preprocess_messages(input)

        # Call the parent ainvoke method with processed input
        return await super().ainvoke(processed_input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        """Stream the model with input that can be messages or raw message dictionaries.

        Args:
            input: Can be a list of BaseMessage objects or raw message dictionaries
            config: Configuration for the invocation
            **kwargs: Additional arguments

        Returns:
            Iterator of ChatGenerationChunk objects
        """
        # Preprocess input to ensure compatibility
        processed_input = self._preprocess_messages(input)

        # Call the parent stream method with processed input
        return super().stream(processed_input, config=config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        """Async stream the model with input that can be messages or raw message dictionaries.

        Args:
            input: Can be a list of BaseMessage objects or raw message dictionaries
            config: Configuration for the invocation
            **kwargs: Additional arguments

        Returns:
            AsyncIterator of ChatGenerationChunk objects
        """
        # Preprocess input to ensure compatibility
        processed_input = self._preprocess_messages(input)

        # Call the parent astream method with processed input and yield chunks
        async for chunk in super().astream(processed_input, config=config, **kwargs):
            yield chunk

    def create_multimodal_message(self, text: str, image_base64: str, detail: str = "auto") -> dict:
        """Create a multimodal message with text and image content.

        Args:
            text: The text content
            image_base64: Base64 encoded image (with or without data URL prefix)
            detail: Image detail level ("auto", "low", "high")

        Returns:
            A message dictionary ready for use with the model
        """
        # Ensure the image_base64 has the proper data URL format if it doesn't already
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"

        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"source_type": "base64", "url": image_base64},
                    "detail": detail,
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }

    def create_multimodal_human_message(self, text: str, image_base64: str, detail: str = "auto") -> HumanMessage:
        """Create a multimodal HumanMessage with text and image content.

        This method provides full LangChain compatibility by returning a proper BaseMessage object.

        Args:
            text: The text content
            image_base64: Base64 encoded image (with or without data URL prefix)
            detail: Image detail level ("auto", "low", "high")

        Returns:
            A HumanMessage object with multimodal content
        """
        # Ensure the image_base64 has the proper data URL format if it doesn't already
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"

        multimodal_content: str | list[str | dict[Any, Any]] = [
            {
                "type": "image_url",
                "image_url": {"source_type": "base64", "url": image_base64},
                "detail": detail,
            },
            {
                "type": "text",
                "text": text,
            },
        ]

        return HumanMessage(content=multimodal_content)


class _StructuredOutputParser:
    """Parser for structured output from SecureGPT."""

    def __init__(
        self,
        llm: SecureGPTChat,
        schema: Union[Dict, Type[BaseModel], type, None],
        include_raw: bool = False,
    ):
        self.llm = llm
        self.schema = schema
        self.include_raw = include_raw

    def invoke(self, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> Any:
        """Invoke the model and parse the structured output."""
        import json

        # Convert string input to list of BaseMessage objects
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        # Add instructions for JSON output
        if messages and messages[-1].content:
            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                # Cast to BaseModel type to access model_json_schema
                basemodel_schema = cast(Type[BaseModel], self.schema)
                schema_str = json.dumps(basemodel_schema.model_json_schema(), indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            elif self.schema is not None:
                schema_str = json.dumps(self.schema, indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            else:
                instruction = "\n\nPlease respond with valid JSON."

            # Ensure content is a string before concatenation
            if isinstance(messages[-1].content, str):
                messages[-1].content += instruction
            else:
                # Handle list content by converting to string representation
                messages[-1].content = str(messages[-1].content) + instruction

        response = self.llm.invoke(messages, **kwargs)

        try:
            parsed_content = json.loads(response.content)

            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                structured_output = self.schema(**parsed_content)
            else:
                structured_output = parsed_content

            if self.include_raw:
                return {
                    "raw": response,
                    "parsed": structured_output,
                }
            else:
                return structured_output

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if self.include_raw:
                return {
                    "raw": response,
                    "parsed": None,
                    "parsing_error": str(e),
                }
            else:
                raise ValueError(f"Failed to parse structured output: {e}") from e

    async def ainvoke(self, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> Any:
        """Async invoke the model and parse the structured output."""
        import json

        # Convert string input to list of BaseMessage objects
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        # Add instructions for JSON output
        if messages and messages[-1].content:
            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                # Cast to BaseModel type to access model_json_schema
                basemodel_schema = cast(Type[BaseModel], self.schema)
                schema_str = json.dumps(basemodel_schema.model_json_schema(), indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            elif self.schema is not None:
                schema_str = json.dumps(self.schema, indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            else:
                instruction = "\n\nPlease respond with valid JSON."

            # Ensure content is a string before concatenation
            if isinstance(messages[-1].content, str):
                messages[-1].content += instruction
            else:
                # Handle list content by converting to string representation
                messages[-1].content = str(messages[-1].content) + instruction

        response = await self.llm.ainvoke(messages, **kwargs)

        try:
            parsed_content = json.loads(response.content)

            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                structured_output = self.schema(**parsed_content)
            else:
                structured_output = parsed_content

            if self.include_raw:
                return {
                    "raw": response,
                    "parsed": structured_output,
                }
            else:
                return structured_output

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if self.include_raw:
                return {
                    "raw": response,
                    "parsed": None,
                    "parsing_error": str(e),
                }
            else:
                raise ValueError(f"Failed to parse structured output: {e}") from e

    def stream(self, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> Iterator[Any]:
        """Stream the model and parse the structured output."""
        import json

        # Convert string input to list of BaseMessage objects
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        # Add instructions for JSON output
        if messages and messages[-1].content:
            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                # Cast to BaseModel type to access model_json_schema
                basemodel_schema = cast(Type[BaseModel], self.schema)
                schema_str = json.dumps(basemodel_schema.model_json_schema(), indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            elif self.schema is not None:
                schema_str = json.dumps(self.schema, indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            else:
                instruction = "\n\nPlease respond with valid JSON."

            # Ensure content is a string before concatenation
            if isinstance(messages[-1].content, str):
                messages[-1].content += instruction
            else:
                # Handle list content by converting to string representation
                messages[-1].content = str(messages[-1].content) + instruction

        # Collect all chunks
        full_content = ""
        for chunk in self.llm.stream(messages, **kwargs):
            full_content += chunk.content
            yield chunk

        # Parse the final result
        try:
            parsed_content = json.loads(full_content)

            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                structured_output = self.schema(**parsed_content)
            else:
                structured_output = parsed_content

            # Yield final structured result
            yield structured_output

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if self.include_raw:
                yield {
                    "raw": full_content,
                    "parsed": None,
                    "parsing_error": str(e),
                }
            else:
                raise

    async def astream(self, messages: Union[str, list[BaseMessage]], **kwargs: Any) -> AsyncIterator[Any]:
        """Async stream the model and parse the structured output."""
        import json

        # Convert string input to list of BaseMessage objects
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        # Add instructions for JSON output
        if messages and messages[-1].content:
            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                # Cast to BaseModel type to access model_json_schema
                basemodel_schema = cast(Type[BaseModel], self.schema)
                schema_str = json.dumps(basemodel_schema.model_json_schema(), indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            elif self.schema is not None:
                schema_str = json.dumps(self.schema, indent=2)
                instruction = f"\n\nPlease respond with valid JSON that matches this schema:\n{schema_str}"
            else:
                instruction = "\n\nPlease respond with valid JSON."

            # Ensure content is a string before concatenation
            if isinstance(messages[-1].content, str):
                messages[-1].content += instruction
            else:
                # Handle list content by converting to string representation
                messages[-1].content = str(messages[-1].content) + instruction

        # Collect all chunks
        full_content = ""
        async for chunk in self.llm.astream(messages, **kwargs):
            full_content += chunk.content
            yield chunk

        # Parse the final result
        try:
            parsed_content = json.loads(full_content)

            if isinstance(self.schema, type) and is_basemodel_subclass(self.schema):
                structured_output = self.schema(**parsed_content)
            else:
                structured_output = parsed_content

            # Yield final structured result
            yield structured_output

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if self.include_raw:
                yield {
                    "raw": full_content,
                    "parsed": None,
                    "parsing_error": str(e),
                }
            else:
                raise ValueError(f"Failed to parse structured output: {e}") from e
