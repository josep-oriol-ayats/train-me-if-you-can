"""Test cases for SecureGptEmbeddings class."""

import os
import warnings
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cai_securegpt_client.onelogin_auth import OneloginAuth
from cai_securegpt_client.securegpt_embeddings import SecureGptEmbeddings, generate_url

# Suppress Pydantic deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


class TestSecureGptEmbeddings:
    """Test cases for SecureGptEmbeddings class."""

    @pytest.fixture
    def mock_auth_manager(self):
        """Mock auth manager for testing."""
        # Create a real OneloginAuth instance but mock its methods
        auth_manager = OneloginAuth()
        auth_manager.get_access_token = Mock(return_value="test_token")
        auth_manager.get_access_token_async = AsyncMock(return_value="test_token")
        return auth_manager

    @pytest.fixture
    def embeddings_model(self, mock_auth_manager):
        """Create a SecureGptEmbeddings instance for testing."""
        with patch.dict(
            os.environ,
            {
                "SECUREGPT_URL": "https://test.example.com",
                "SECUREGPT_EMBEDDINGS_MODEL_ID": "text-embedding-ada-002",
                "SECUREGPT_EMBEDDINGS_PROVIDER": "openai",
                "SECUREGPT_TIMEOUT": "60",
                "SECUREGPT_OPENAI_API_VERSION": "2024-10-21",
            },
        ):
            # Use patch to avoid Pydantic validation issues
            with patch(
                "cai_securegpt_client.securegpt_embeddings.OneloginAuth",
                return_value=mock_auth_manager,
            ):
                model = SecureGptEmbeddings()
                model.auth_manager = mock_auth_manager
                return model

    @pytest.fixture
    def embeddings_model_no_auth(self):
        """Create a SecureGptEmbeddings instance without auth manager for testing."""
        with patch.dict(
            os.environ,
            {
                "SECUREGPT_URL": "https://test.example.com",
                "SECUREGPT_EMBEDDINGS_MODEL_ID": "text-embedding-ada-002",
                "SECUREGPT_EMBEDDINGS_PROVIDER": "openai",
                "SECUREGPT_TIMEOUT": "60",
                "SECUREGPT_OPENAI_API_VERSION": "2024-10-21",
            },
        ):
            model = SecureGptEmbeddings(auth_manager=None, secure_gpt_access_token="direct_token")
            return model

    @pytest.fixture
    def mock_embedding_response(self):
        """Mock embedding response from API."""
        return {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0,
                    "object": "embedding",
                }
            ],
            "model": "text-embedding-ada-002",
            "object": "list",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @pytest.fixture
    def mock_successful_response(self, mock_embedding_response):
        """Mock successful HTTP response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_embedding_response
        return mock_response

    @pytest.fixture
    def mock_error_response(self):
        """Mock error HTTP response."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad Request"}
        return mock_response

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        with patch.dict(
            os.environ,
            {
                "SECUREGPT_URL": "https://test.example.com",
                "SECUREGPT_EMBEDDINGS_MODEL_ID": "test-model",
                "SECUREGPT_EMBEDDINGS_PROVIDER": "openai",
                "SECUREGPT_TIMEOUT": "60",
                "SECUREGPT_OPENAI_API_VERSION": "2024-10-21",
            },
            clear=True,
        ):
            embeddings = SecureGptEmbeddings()
            assert embeddings.model_id == "test-model"
            assert embeddings.provider == "openai"
            assert embeddings.secure_gpt_api_base == "https://test.example.com"
            assert embeddings.timeout == 60
            assert embeddings.client is not None
            assert embeddings.async_client is not None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        embeddings = SecureGptEmbeddings(
            model_id="custom-model",
            provider="mistral",
            secure_gpt_api_base="https://custom.example.com",
            timeout=120,
            secure_gpt_access_token="custom_token",
        )

        assert embeddings.model_id == "custom-model"
        assert embeddings.provider == "mistral"
        assert embeddings.secure_gpt_api_base == "https://custom.example.com"
        assert embeddings.timeout == 120
        assert embeddings.secure_gpt_access_token == "custom_token"

    def test_prepare_input_with_token(self, embeddings_model_no_auth):
        """Test _prepare_input method with access token."""
        input_body, headers = embeddings_model_no_auth._prepare_input("test text")

        assert input_body == {"input": "test text"}
        assert headers == {"Authorization": "Bearer direct_token"}

    def test_prepare_input_with_auth_manager(self, embeddings_model):
        """Test _prepare_input method with auth manager."""
        embeddings_model.secure_gpt_access_token = None

        input_body, headers = embeddings_model._prepare_input("test text")

        assert input_body == {"input": "test text"}
        assert headers == {"Authorization": "Bearer test_token"}

    def test_prepare_input_text_cleanup(self, embeddings_model):
        """Test _prepare_input method cleans up line separators."""
        test_text = f"line1{os.linesep}line2{os.linesep}line3"

        input_body, headers = embeddings_model._prepare_input(test_text)

        assert input_body == {"input": "line1 line2 line3"}

    @patch("httpx.Client.post")
    def test_embedding_func_success(self, mock_post, embeddings_model, mock_successful_response):
        """Test _embedding_func method with successful response."""
        mock_post.return_value = mock_successful_response

        result = embeddings_model._embedding_func("test text")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    @patch("httpx.Client.post")
    def test_embedding_func_error_response(self, mock_post, embeddings_model, mock_error_response):
        """Test _embedding_func method with error response."""
        mock_post.return_value = mock_error_response

        with pytest.raises(ValueError, match="Error invoking secureGPT embeddings endpoint"):
            embeddings_model._embedding_func("test text")

    @patch("httpx.Client.post")
    def test_embedding_func_exception(self, mock_post, embeddings_model):
        """Test _embedding_func method with exception."""
        mock_post.side_effect = Exception("Network error")

        with pytest.raises(ValueError, match="Error raised by inference endpoint"):
            embeddings_model._embedding_func("test text")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembedding_func_success(self, mock_post, embeddings_model, mock_successful_response):
        """Test _aembedding_func method with successful response."""
        mock_post.return_value = mock_successful_response

        result = await embeddings_model._aembedding_func("test text")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembedding_func_error_response(self, mock_post, embeddings_model, mock_error_response):
        """Test _aembedding_func method with error response."""
        mock_post.return_value = mock_error_response

        with pytest.raises(ValueError, match="Error invoking secureGPT embeddings endpoint"):
            await embeddings_model._aembedding_func("test text")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembedding_func_exception(self, mock_post, embeddings_model):
        """Test _aembedding_func method with exception."""
        mock_post.side_effect = Exception("Network error")

        with pytest.raises(ValueError, match="Error raised by inference endpoint"):
            await embeddings_model._aembedding_func("test text")

    @patch("httpx.Client.post")
    def test_embed_query(self, mock_post, embeddings_model, mock_successful_response):
        """Test embed_query method."""
        mock_post.return_value = mock_successful_response

        result = embeddings_model.embed_query("test query")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    @patch("httpx.Client.post")
    def test_embed_documents(self, mock_post, embeddings_model, mock_successful_response):
        """Test embed_documents method."""
        mock_post.return_value = mock_successful_response

        texts = ["doc1", "doc2", "doc3"]
        results = embeddings_model.embed_documents(texts)

        assert len(results) == 3
        assert all(result == [0.1, 0.2, 0.3, 0.4, 0.5] for result in results)
        assert mock_post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembed_query(self, mock_post, embeddings_model, mock_successful_response):
        """Test aembed_query method."""
        mock_post.return_value = mock_successful_response

        result = await embeddings_model.aembed_query("test query")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembed_documents(self, mock_post, embeddings_model, mock_successful_response):
        """Test aembed_documents method."""
        mock_post.return_value = mock_successful_response

        texts = ["doc1", "doc2", "doc3"]
        results = await embeddings_model.aembed_documents(texts)

        assert len(results) == 3
        assert all(result == [0.1, 0.2, 0.3, 0.4, 0.5] for result in results)
        assert mock_post.call_count == 3

    @patch("httpx.Client.post")
    def test_embed_documents_empty_list(self, mock_post, embeddings_model):
        """Test embed_documents with empty list."""
        results = embeddings_model.embed_documents([])

        assert results == []
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_aembed_documents_empty_list(self, mock_post, embeddings_model):
        """Test aembed_documents with empty list."""
        results = await embeddings_model.aembed_documents([])

        assert results == []
        mock_post.assert_not_called()

    def test_langchain_interface_compliance(self, embeddings_model):
        """Test that the class implements required Langchain Embeddings interface."""
        from langchain_core.embeddings import Embeddings

        assert isinstance(embeddings_model, Embeddings)
        assert hasattr(embeddings_model, "embed_documents")
        assert hasattr(embeddings_model, "embed_query")
        assert hasattr(embeddings_model, "aembed_documents")
        assert hasattr(embeddings_model, "aembed_query")

        # Check method signatures
        import inspect

        embed_docs_sig = inspect.signature(embeddings_model.embed_documents)
        assert "texts" in embed_docs_sig.parameters

        embed_query_sig = inspect.signature(embeddings_model.embed_query)
        assert "text" in embed_query_sig.parameters

    @patch("httpx.Client.post")
    def test_get_embedding_dimensions_from_api(self, mock_post, embeddings_model, mock_successful_response):
        """Test get_embedding_dimensions method gets dimensions from actual API call."""
        mock_post.return_value = mock_successful_response

        dimensions = embeddings_model.get_embeddings_dimensions()

        # Should return the length of the mocked embedding response
        assert dimensions == 5  # Length of [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    def test_get_embedder_returns_new_instance(self, embeddings_model):
        """Test get_embedder method returns a new SecureGptEmbeddings instance."""
        embedder = embeddings_model.get_embedder()

        assert isinstance(embedder, SecureGptEmbeddings)
        assert embedder is not embeddings_model  # Should be a different instance
        assert embedder.model_id == embeddings_model.model_id
        assert embedder.provider == embeddings_model.provider
        assert embedder.openai_apiver == embeddings_model.openai_apiver
        assert embedder.secure_gpt_api_base == embeddings_model.secure_gpt_api_base

    def test_get_embedder_with_custom_values(self):
        """Test get_embedder method with custom configuration values."""
        original_embeddings = SecureGptEmbeddings(
            model_id="custom-model",
            provider="mistral",
            openai_apiver="2023-05-15",
            secure_gpt_api_base="https://custom.example.com",
            secure_gpt_access_token="test_token",
        )

        embedder = original_embeddings.get_embedder()

        assert embedder.model_id == "custom-model"
        assert embedder.provider == "mistral"
        assert embedder.openai_apiver == "2023-05-15"
        assert embedder.secure_gpt_api_base == "https://custom.example.com"
        # Note: secure_gpt_access_token is not copied (by design)
        assert embedder.secure_gpt_access_token is None

    def test_get_embedder_independence(self, embeddings_model):
        """Test that get_embedder creates independent instances."""
        embedder1 = embeddings_model.get_embedder()
        embedder2 = embeddings_model.get_embedder()

        assert embedder1 is not embedder2  # Different instances
        assert embedder1.model_id == embedder2.model_id
        assert embedder1.provider == embedder2.provider

    @patch("httpx.Client.post")
    def test_get_embedder_functionality(self, mock_post, embeddings_model, mock_successful_response):
        """Test that embedder returned by get_embedder is fully functional."""
        mock_post.return_value = mock_successful_response

        embedder = embeddings_model.get_embedder()

        # Test that the new embedder can perform embeddings
        result = embedder.embed_query("test query")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

    def test_get_embedding_dimensions_with_different_providers(self):
        """Test get_embedding_dimensions with different providers."""
        # Test with OpenAI provider
        openai_embeddings = SecureGptEmbeddings(
            provider="openai",
            model_id="text-embedding-ada-002",
            secure_gpt_access_token="test_token",
        )

        # Test with Mistral provider
        mistral_embeddings = SecureGptEmbeddings(
            provider="mistral",
            model_id="mistral-embed",
            secure_gpt_access_token="test_token",
        )

        with patch("httpx.Client.post") as mock_post:
            # Mock different embedding sizes for different providers
            openai_response = Mock()
            openai_response.status_code = 200
            openai_response.json.return_value = {"data": [{"embedding": [0.1] * 1536}]}

            mistral_response = Mock()
            mistral_response.status_code = 200
            mistral_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}

            # Configure different responses for different calls
            mock_post.side_effect = [openai_response, mistral_response]

            openai_dims = openai_embeddings.get_embeddings_dimensions()
            mistral_dims = mistral_embeddings.get_embeddings_dimensions()

            assert openai_dims == 1536
            assert mistral_dims == 1024


class TestGenerateUrl:
    """Test cases for generate_url function."""

    def test_generate_url_openai(self):
        """Test URL generation for OpenAI provider."""
        url = generate_url("https://test.example.com", "openai", "text-embedding-ada-002", "2024-10-21")

        expected = "https://test.example.com/providers/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21"
        assert url == expected

    def test_generate_url_non_openai(self):
        """Test URL generation for non-OpenAI provider."""
        url = generate_url("https://test.example.com", "mistral", "mistral-embed", "2024-10-21")

        expected = "https://test.example.com/providers/mistral/deployments/mistral-embed/embeddings"
        assert url == expected

    def test_generate_url_default_api_version(self):
        """Test URL generation with default API version."""
        url = generate_url("https://test.example.com", "openai", "text-embedding-ada-002")

        expected = "https://test.example.com/providers/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-10-21"
        assert url == expected


class TestSecureGptEmbeddingsIntegration:
    """Integration tests with real credentials (requires environment setup)."""

    @pytest.fixture
    def real_embeddings_model(self):
        """Create a real SecureGptEmbeddings instance for integration testing."""
        # Check for any available credential method
        has_direct_token = os.getenv("SECUREGPT_ACCESS_TOKEN")
        has_onelogin = os.getenv("ONELOGIN_CLIENT_ID") and os.getenv("ONELOGIN_CLIENT_SECRET")
        has_url = os.getenv("SECUREGPT_URL")

        # Skip if no credentials are available at all
        if not (has_direct_token or has_onelogin) and not has_url:
            pytest.skip(
                "Real credentials not available for integration testing. "
                "Set SECUREGPT_ACCESS_TOKEN or ONELOGIN_CLIENT_ID/ONELOGIN_CLIENT_SECRET "
                "and optionally SECUREGPT_URL"
            )

        # Create model with available credentials
        kwargs = {}
        if has_direct_token:
            kwargs["secure_gpt_access_token"] = os.getenv("SECUREGPT_ACCESS_TOKEN")
        if has_url:
            kwargs["secure_gpt_api_base"] = os.getenv("SECUREGPT_URL")

        return SecureGptEmbeddings(**kwargs)

    @pytest.mark.integration
    def test_real_embed_query(self, real_embeddings_model):
        """Test embed_query with real credentials."""
        result = real_embeddings_model.embed_query("Hello, world!")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.integration
    def test_real_embed_documents(self, real_embeddings_model):
        """Test embed_documents with real credentials."""
        texts = ["Hello, world!", "This is a test document.", "Another test text."]
        results = real_embeddings_model.embed_documents(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
        assert all(len(result) > 0 for result in results)
        assert all(isinstance(x, float) for result in results for x in result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_aembed_query(self, real_embeddings_model):
        """Test aembed_query with real credentials."""
        result = await real_embeddings_model.aembed_query("Hello, world!")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_aembed_documents(self, real_embeddings_model):
        """Test aembed_documents with real credentials."""
        texts = ["Hello, world!", "This is a test document.", "Another test text."]
        results = await real_embeddings_model.aembed_documents(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
        assert all(len(result) > 0 for result in results)
        assert all(isinstance(x, float) for result in results for x in result)

    @pytest.mark.integration
    def test_real_error_handling(self, real_embeddings_model):
        """Test error handling with real credentials and invalid input."""
        # Test with extremely long text that might cause issues
        very_long_text = "test " * 10000

        try:
            result = real_embeddings_model.embed_query(very_long_text)
            # If it succeeds, verify the result is valid
            assert isinstance(result, list)
            assert len(result) > 0
        except ValueError as e:
            # If it fails, verify it's a proper error
            assert "Error" in str(e)

    @pytest.mark.integration
    def test_real_different_providers(self):
        """Test with different providers if available."""
        providers = ["openai", "mistral"]

        for provider in providers:
            if os.getenv(f"SECUREGPT_{provider.upper()}_MODEL_ID"):
                embeddings = SecureGptEmbeddings(
                    provider=provider,
                    model_id=os.getenv(f"SECUREGPT_{provider.upper()}_MODEL_ID"),
                    secure_gpt_access_token=os.getenv("SECUREGPT_ACCESS_TOKEN"),
                )

                result = embeddings.embed_query("Test with " + provider)
                assert isinstance(result, list)
                assert len(result) > 0

    @pytest.mark.integration
    def test_real_get_embedder(self, real_embeddings_model):
        """Test get_embedder with real credentials."""
        embedder = real_embeddings_model.get_embedder()

        assert isinstance(embedder, SecureGptEmbeddings)
        assert embedder is not real_embeddings_model  # Different instance
        assert embedder.model_id == real_embeddings_model.model_id
        assert embedder.provider == real_embeddings_model.provider

        # Test that the new embedder can perform embeddings
        result = embedder.embed_query("Test embedder functionality")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.integration
    def test_real_get_embedding_dimensions(self, real_embeddings_model):
        """Test get_embedding_dimensions with real credentials."""
        dimensions = real_embeddings_model.get_embeddings_dimensions()

        assert isinstance(dimensions, int)
        assert dimensions > 0
        # Common embedding dimensions
        assert dimensions in [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

    @pytest.mark.integration
    def test_real_get_embedding_dimensions_consistency(self, real_embeddings_model):
        """Test that get_embedding_dimensions returns consistent results."""
        # Call multiple times to ensure consistency
        dim1 = real_embeddings_model.get_embeddings_dimensions()
        dim2 = real_embeddings_model.get_embeddings_dimensions()
        dim3 = real_embeddings_model.get_embeddings_dimensions()

        assert dim1 == dim2 == dim3
        assert isinstance(dim1, int)
        assert dim1 > 0

    @pytest.mark.integration
    def test_real_embedding_dimensions_match_actual_output(self, real_embeddings_model):
        """Test that get_embedding_dimensions matches actual embedding output length."""
        # Get dimensions using the method
        dimensions = real_embeddings_model.get_embeddings_dimensions()

        # Get an actual embedding
        embedding = real_embeddings_model.embed_query("Test embedding length")

        # They should match
        assert len(embedding) == dimensions

    @pytest.mark.integration
    def test_real_get_embedder_with_different_configs(self):
        """Test get_embedder with different model configurations."""
        # Test with different providers if available
        providers_models = [
            ("openai", "text-embedding-ada-002"),
            ("mistral", "mistral-embed"),
        ]

        for provider, model_id in providers_models:
            # Skip if credentials not available
            if not os.getenv("SECUREGPT_ACCESS_TOKEN"):
                continue

            try:
                original_embeddings = SecureGptEmbeddings(
                    provider=provider,
                    model_id=model_id,
                    secure_gpt_access_token=os.getenv("SECUREGPT_ACCESS_TOKEN"),
                )

                embedder = original_embeddings.get_embedder()

                # Test basic functionality
                result = embedder.embed_query("Test with " + provider)
                assert isinstance(result, list)
                assert len(result) > 0

                # Test configuration is preserved
                assert embedder.provider == provider
                assert embedder.model_id == model_id

            except Exception as e:
                # Skip if model/provider not available
                if "not found" in str(e) or "not available" in str(e):
                    continue
                raise


class TestSecureGptEmbeddingsPerformance:
    """Performance tests for SecureGptEmbeddings."""

    @pytest.fixture
    def performance_embeddings_model(self):
        """Create embeddings model for performance testing."""
        return SecureGptEmbeddings(secure_gpt_access_token="test_token")

    @patch("httpx.Client.post")
    def test_batch_processing_performance(self, mock_post, performance_embeddings_model, mock_successful_response):
        """Test performance of batch processing."""
        mock_post.return_value = mock_successful_response

        # Test with a reasonable batch size
        texts = [f"Document {i}" for i in range(10)]

        import time

        start_time = time.time()
        results = performance_embeddings_model.embed_documents(texts)
        end_time = time.time()

        assert len(results) == 10
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds with mocked responses

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_async_batch_processing_performance(
        self, mock_post, performance_embeddings_model, mock_successful_response
    ):
        """Test performance of async batch processing."""
        mock_post.return_value = mock_successful_response

        # Test with a reasonable batch size
        texts = [f"Document {i}" for i in range(10)]

        import time

        start_time = time.time()
        results = await performance_embeddings_model.aembed_documents(texts)
        end_time = time.time()

        assert len(results) == 10
        assert (end_time - start_time) < 2.0  # Async should be faster than sync

    def test_memory_usage_with_large_texts(self, performance_embeddings_model):
        """Test memory usage with large text inputs."""
        # Create a large text
        large_text = "This is a test sentence. " * 1000

        # Monitor memory usage (basic check)
        import tracemalloc

        tracemalloc.start()

        try:
            with patch("httpx.Client.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                # Use dynamic embedding dimensions instead of hardcoded 1536
                from cai_securegpt_client.securegpt_embeddings import (
                    _get_embedding_dimensions,
                )

                dimensions = _get_embedding_dimensions()
                mock_response.json.return_value = {"data": [{"embedding": [0.1] * dimensions}]}
                mock_post.return_value = mock_response

                result = performance_embeddings_model.embed_query(large_text)
                assert isinstance(result, list)

        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Basic memory check - should not use excessive memory
            assert peak < 100 * 1024 * 1024  # Less than 100MB

    @pytest.fixture
    def mock_successful_response(self):
        """Mock successful response for performance tests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}
        return mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
