import asyncio
import os
import time
from typing import Any, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from .onelogin_auth import OneloginAuth  # type: ignore

load_dotenv()


def _get_embedding_dimensions() -> int:
    """Get the embedding dimensions from environment variable."""
    return int(os.getenv("SECUREGPT_EMBEDDINGS_DIMENSIONS", "1536"))


def generate_url(base_url: str, provider: str, model_id: str, openai_api_ver: str = "2024-10-21") -> str:
    """Generates the url to call the secure gpt embeddings API.

    Arguments:
        base_url -- The secure gpt hub base url
        provider -- The model provider (openai, mistral...)
        model_id -- The model id (https://pages.github.axa.com/axa-go-getd/secure-gpt-model-hub/offerings/models/)

    Keyword Arguments:
        openai_api_ver -- openai API version (default: {"2024-10-21"})
        is_stream -- if the url is for streaming (default: {False})

    Returns:
        The url
    """
    api_version_param = f"?api-version={openai_api_ver}" if provider == "openai" else ""
    url = f"{base_url}/providers/{provider}/deployments/{model_id}/embeddings{api_version_param}"
    return url


class SecureGptEmbeddings(BaseModel, Embeddings):
    """Embeddings provider for SecureGPT models."""

    auth_manager: Optional[OneloginAuth] = None
    model_id: str = Field(
        default_factory=lambda: os.getenv("SECUREGPT_EMBEDDINGS_MODEL_ID", "text-embedding-ada-002-2")
    )
    provider: str = Field(default_factory=lambda: os.getenv("SECUREGPT_EMBEDDINGS_PROVIDER", "openai"))
    openai_apiver: str = Field(default_factory=lambda: os.getenv("SECUREGPT_OPENAI_API_VERSION", "2024-10-21"))
    secure_gpt_api_base: str = Field(
        default_factory=lambda: os.getenv(
            "SECUREGPT_URL",
            "https://api.se.axa-go.applications.services.axa-tech.intraxa/ago-m365-securegpt-hub-v1-vrs",
        )
    )
    secure_gpt_access_token: Optional[str] = None
    timeout: int = Field(default_factory=lambda: int(os.getenv("SECUREGPT_TIMEOUT", "60")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("SECUREGPT_MAX_RETRIES", "2")))

    # Add retry configuration fields
    retry_on_status_codes: List[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_backoff_factor: float = Field(default=2.0)
    retry_jitter: bool = Field(default=True)

    client: httpx.Client | None = None
    async_client: httpx.AsyncClient | None = None

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize the SecureGptEmbeddings with HTTP clients."""
        # Set default auth_manager if not provided
        if "auth_manager" not in data and not data.get("secure_gpt_access_token"):
            data["auth_manager"] = OneloginAuth()

        super().__init__(**data)
        self.client = httpx.Client(
            base_url=self.secure_gpt_api_base,
            timeout=self.timeout,
            verify=False,
        )
        self.async_client = httpx.AsyncClient(
            base_url=self.secure_gpt_api_base,
            timeout=self.timeout,
            verify=False,
        )

    def _prepare_input(self, text: str) -> Tuple[dict, dict]:
        """Prepare input for the embedding and headers.

        Arguments:
            text -- The input text

        Returns:
            A dict with the input body and another with the auth headers.
        """
        text = text.replace(os.linesep, " ")

        # format input body for provider
        input_body = {"input": text}
        # invoke secureGPT API
        api_key = self.secure_gpt_access_token
        if self.auth_manager:
            api_key = self.auth_manager.get_access_token()

        headers = {"Authorization": f"Bearer {api_key}"}
        return input_body, headers

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

    def _embedding_func(self, text: str) -> List[float]:
        """Internal method to call the embeddings API synchronously with retry logic.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats.
        """
        input_body, headers = self._prepare_input(text)
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # invoke secureGPT API
                if self.client:
                    response = self.client.post(
                        generate_url(
                            self.secure_gpt_api_base,
                            self.provider,
                            self.model_id,
                            self.openai_apiver,
                        ),
                        json=input_body,
                        headers=headers,
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

                    # format output based on provider
                    response_body = response.json()
                    return response_body.get("data")[0].get("embedding")
                return []

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                else:
                    # On final attempt, raise the exception
                    if hasattr(e, "response") and e.response is not None:
                        raise ValueError(f"Error invoking secureGPT embeddings endpoint: {e}") from e
                    else:
                        raise ValueError(f"Network error calling secureGPT embeddings: {e}") from e
            except Exception as e:
                # For non-retryable exceptions, raise immediately
                raise ValueError(f"Unexpected error calling secureGPT embeddings: {e}") from e

        # This should never be reached, but just in case
        if last_exception:
            raise ValueError(
                f"Error invoking secureGPT embeddings endpoint after {self.max_retries} retries"
            ) from last_exception
        return []

    async def _aembedding_func(self, text: str) -> List[float]:
        """Internal method to call the embeddings API asynchronously with retry logic.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats.
        """
        input_body, headers = self._prepare_input(text)
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # invoke secureGPT API
                if self.async_client:
                    response = await self.async_client.post(
                        generate_url(
                            self.secure_gpt_api_base,
                            self.provider,
                            self.model_id,
                            self.openai_apiver,
                        ),
                        json=input_body,
                        headers=headers,
                    )

                    # Check if we should retry based on status code
                    if response.status_code in self.retry_on_status_codes and attempt < self.max_retries:
                        delay = self._exponential_backoff(attempt)
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: {response.text}",
                            request=response.request,
                            response=response,
                        )

                    # format output based on provider
                    response_body = response.json()
                    return response_body.get("data")[0].get("embedding")
                return []

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._exponential_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    # On final attempt, raise the exception
                    if hasattr(e, "response") and e.response is not None:
                        raise ValueError(f"Error invoking secureGPT embeddings endpoint: {e}") from e
                    else:
                        raise ValueError(f"Network error calling secureGPT embeddings: {e}") from e
            except Exception as e:
                # For non-retryable exceptions, raise immediately
                raise ValueError(f"Unexpected error calling secureGPT embeddings: {e}") from e

        # This should never be reached, but just in case
        if last_exception:
            raise ValueError(
                f"Error invoking secureGPT embeddings endpoint after {self.max_retries} retries"
            ) from last_exception
        return []

    def get_embeddings_dimensions(self) -> int:
        """Get the embedding dimensions dynamically.

        Returns:
            The number of dimensions in the embeddings.
        """
        try:
            # Try to get dimensions from a small test embedding
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception:
            raise

    def get_embedder(self) -> Any:  # noqa: ANN401
        """Get the Azure OpenAI embedder instance."""
        return SecureGptEmbeddings(
            model_id=self.model_id,
            provider=self.provider,
            openai_apiver=self.openai_apiver,
            secure_gpt_api_base=self.secure_gpt_api_base,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        for text in texts:
            response = self._embedding_func(text)
            results.append(response)

        return results

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Bedrock model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embedding_func(text)

        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous compute query embeddings using a Bedrock model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return await self._aembedding_func(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings, one for each text.
        """
        result = await asyncio.gather(*[self._aembedding_func(text) for text in texts])

        return list(result)
