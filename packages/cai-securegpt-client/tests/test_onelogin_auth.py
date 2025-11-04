import os
import warnings

import pytest

from cai_securegpt_client.onelogin_auth import OneloginAuth

# Suppress Pydantic deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


class MockToken:
    """Mock token class for testing with call tracking."""

    def __init__(self, access_token, is_expired=False):
        """Initialize mock token with access token and expiration status."""
        self.access_token = access_token
        self._is_expired = is_expired
        self.is_expired_call_count = 0

    def is_expired(self):
        """Check if token is expired and track call count."""
        self.is_expired_call_count += 1
        return self._is_expired


class MockOAuth2Client:
    """Mock OAuth2Client class for testing with comprehensive tracking."""

    def __init__(self, token_endpoint, auth):
        """Initialize mock OAuth2Client with token endpoint and auth credentials."""
        self.token_endpoint = token_endpoint
        self.auth = auth
        self.call_count = 0
        self.client_credentials_call_count = 0
        self.client_credentials_scope = None
        self.return_token = None
        self.should_raise = None

    def client_credentials(self, scope=None):
        """Mock client credentials method with error handling."""
        self.client_credentials_call_count += 1
        self.client_credentials_scope = scope
        if self.should_raise:
            raise self.should_raise
        return self.return_token


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing."""
    return {
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "token_endpoint": "https://test.example.com/token",
    }


@pytest.fixture
def mock_token():
    """Mock valid token for testing."""
    return MockToken("test_access_token", is_expired=False)


@pytest.fixture
def expired_token():
    """Mock expired token for testing."""
    return MockToken("expired_token", is_expired=True)


class TestOneloginAuth:
    """Test class for OneloginAuth with comprehensive test coverage."""

    def test_init_with_provided_credentials(self, mock_credentials):
        """Test initialization with all credentials provided."""
        auth = OneloginAuth(**mock_credentials)

        assert auth.client_id == mock_credentials["client_id"]
        assert auth.client_secret == mock_credentials["client_secret"]
        assert auth.token_endpoint == mock_credentials["token_endpoint"]
        assert auth.token is None

    def test_init_with_environment_variables(self, monkeypatch):
        """Test initialization using environment variables."""
        env_vars = {
            "SECUREGPT_ONELOGIN_CLIENT_ID": "env_client_id",
            "SECUREGPT_ONELOGIN_SECRET": "env_client_secret",
            "SECUREGPT_ONELOGIN_URL": "https://env.example.com/token",
        }
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        auth = OneloginAuth()

        assert auth.client_id == "env_client_id"
        assert auth.client_secret == "env_client_secret"
        assert auth.token_endpoint == "https://env.example.com/token"

    def test_init_with_default_token_endpoint(self, monkeypatch):
        """Test initialization with default token endpoint when env var is not set."""
        for env_var in [
            "SECUREGPT_ONELOGIN_CLIENT_ID",
            "SECUREGPT_ONELOGIN_SECRET",
            "SECUREGPT_ONELOGIN_URL",
        ]:
            monkeypatch.delenv(env_var, raising=False)

        auth = OneloginAuth()
        assert auth.token_endpoint == "https://onelogin.axa.com/as/token.oauth2"

    def test_init_partial_override(self, monkeypatch):
        """Test initialization with partial credential override."""
        monkeypatch.setenv("SECUREGPT_ONELOGIN_CLIENT_ID", "env_client_id")
        monkeypatch.setenv("SECUREGPT_ONELOGIN_SECRET", "env_client_secret")

        auth = OneloginAuth(client_id="override_client_id")

        assert auth.client_id == "override_client_id"
        assert auth.client_secret == "env_client_secret"

    def _setup_oauth_mock(self, monkeypatch, mock_credentials, return_token):
        """Helper method to setup OAuth2Client mock with consistent behavior."""
        mock_client_instance = MockOAuth2Client(
            token_endpoint=mock_credentials["token_endpoint"],
            auth=(mock_credentials["client_id"], mock_credentials["client_secret"]),
        )
        mock_client_instance.return_token = return_token

        def mock_oauth2_client_constructor(token_endpoint, auth):
            mock_client_instance.token_endpoint = token_endpoint
            mock_client_instance.auth = auth
            mock_client_instance.call_count += 1
            return mock_client_instance

        monkeypatch.setattr(
            "cai_securegpt_client.onelogin_auth.OAuth2Client",
            mock_oauth2_client_constructor,
        )
        return mock_client_instance

    def test_get_access_token_new_token(self, monkeypatch, mock_credentials, mock_token):
        """Test getting access token when no token exists."""
        mock_client_instance = self._setup_oauth_mock(monkeypatch, mock_credentials, mock_token)

        auth = OneloginAuth(**mock_credentials)
        token = auth.get_access_token()

        # Verify OAuth2Client was called correctly
        assert mock_client_instance.call_count == 1
        assert mock_client_instance.token_endpoint == mock_credentials["token_endpoint"]
        assert mock_client_instance.auth == (
            mock_credentials["client_id"],
            mock_credentials["client_secret"],
        )

        # Verify client_credentials was called with correct scope
        assert mock_client_instance.client_credentials_call_count == 1
        assert mock_client_instance.client_credentials_scope == "urn:grp:chatgpt"

        # Verify token was stored and returned
        assert auth.token == mock_token
        assert token == "test_access_token"

    def test_get_access_token_existing_valid_token(self, mock_credentials, mock_token):
        """Test getting access token when valid token already exists."""
        auth = OneloginAuth(**mock_credentials)
        auth.token = mock_token

        token = auth.get_access_token()

        assert token == "test_access_token"
        assert mock_token.is_expired_call_count == 1

    def test_get_access_token_expired_token(self, monkeypatch, mock_credentials, expired_token, mock_token):
        """Test getting access token when existing token is expired."""
        mock_client_instance = self._setup_oauth_mock(monkeypatch, mock_credentials, mock_token)

        auth = OneloginAuth(**mock_credentials)
        auth.token = expired_token

        token = auth.get_access_token()

        # Verify new token was requested
        assert mock_client_instance.call_count == 1
        assert mock_client_instance.client_credentials_call_count == 1
        assert mock_client_instance.client_credentials_scope == "urn:grp:chatgpt"
        assert auth.token == mock_token
        assert token == "test_access_token"

    @pytest.mark.asyncio
    async def test_get_access_token_async_new_token(self, monkeypatch, mock_credentials, mock_token):
        """Test getting access token asynchronously when no token exists."""
        mock_client_instance = self._setup_oauth_mock(monkeypatch, mock_credentials, mock_token)

        auth = OneloginAuth(**mock_credentials)
        token = await auth.get_access_token_async()

        # Verify OAuth2Client was called correctly
        assert mock_client_instance.call_count == 1
        assert mock_client_instance.token_endpoint == mock_credentials["token_endpoint"]
        assert mock_client_instance.auth == (
            mock_credentials["client_id"],
            mock_credentials["client_secret"],
        )

        # Verify client_credentials was called with correct scope
        assert mock_client_instance.client_credentials_call_count == 1
        assert mock_client_instance.client_credentials_scope == "urn:grp:chatgpt"

        # Verify token was stored and returned
        assert auth.token == mock_token
        assert token == "test_access_token"

    @pytest.mark.asyncio
    async def test_get_access_token_async_existing_valid_token(self, mock_credentials, mock_token):
        """Test getting access token asynchronously when valid token already exists."""
        auth = OneloginAuth(**mock_credentials)
        auth.token = mock_token

        token = await auth.get_access_token_async()

        assert token == "test_access_token"
        assert mock_token.is_expired_call_count == 1

    def test_oauth2_client_exception_handling_sync(self, monkeypatch, mock_credentials):
        """Test exception handling when OAuth2Client fails in sync method."""

        def mock_oauth2_client_constructor(token_endpoint, auth):
            raise Exception("OAuth2 error")

        monkeypatch.setattr(
            "cai_securegpt_client.onelogin_auth.OAuth2Client",
            mock_oauth2_client_constructor,
        )

        auth = OneloginAuth(**mock_credentials)

        with pytest.raises(Exception, match="OAuth2 error"):
            auth.get_access_token()

    @pytest.mark.asyncio
    async def test_oauth2_client_exception_handling_async(self, monkeypatch, mock_credentials):
        """Test exception handling when OAuth2Client fails in async method."""

        def mock_oauth2_client_constructor(token_endpoint, auth):
            raise Exception("OAuth2 error")

        monkeypatch.setattr(
            "cai_securegpt_client.onelogin_auth.OAuth2Client",
            mock_oauth2_client_constructor,
        )

        auth = OneloginAuth(**mock_credentials)

        with pytest.raises(Exception, match="OAuth2 error"):
            await auth.get_access_token_async()


class TestOneloginAuthIntegration:
    """Integration tests for OneloginAuth with real environment variables."""

    @pytest.fixture
    def real_env_credentials(self):
        """Get real environment credentials if available."""
        return {
            "client_id": os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
            "client_secret": os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            "token_endpoint": os.getenv("SECUREGPT_ONELOGIN_URL"),
        }

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_init_with_real_environment_credentials(self, real_env_credentials):
        """Test initialization with real environment credentials."""
        auth = OneloginAuth()

        assert auth.client_id == real_env_credentials["client_id"]
        assert auth.client_secret == real_env_credentials["client_secret"]
        assert auth.token_endpoint is not None
        assert auth.token is None

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_with_real_credentials_mocked_client(self, monkeypatch, real_env_credentials):
        """Test with real environment credentials but mocked OAuth2Client."""
        mock_token = MockToken("mocked_real_token", is_expired=False)
        mock_client_instance = MockOAuth2Client(token_endpoint="", auth=("", ""))
        mock_client_instance.return_token = mock_token

        def mock_oauth2_client_constructor(token_endpoint, auth):
            mock_client_instance.token_endpoint = token_endpoint
            mock_client_instance.auth = auth
            mock_client_instance.call_count += 1
            return mock_client_instance

        monkeypatch.setattr(
            "cai_securegpt_client.onelogin_auth.OAuth2Client",
            mock_oauth2_client_constructor,
        )

        auth = OneloginAuth()
        token = auth.get_access_token()

        # Verify OAuth2Client was called with real credentials
        assert mock_client_instance.call_count == 1
        assert mock_client_instance.token_endpoint == auth.token_endpoint
        assert mock_client_instance.auth == (
            real_env_credentials["client_id"],
            real_env_credentials["client_secret"],
        )

        # Verify client_credentials was called with correct scope
        assert mock_client_instance.client_credentials_call_count == 1
        assert mock_client_instance.client_credentials_scope == "urn:grp:chatgpt"

        # Verify token was returned
        assert token == "mocked_real_token"


class TestOneloginAuthRealAuthentication:
    """Real authentication tests that make actual HTTP calls to OneLogin."""

    @pytest.fixture
    def real_auth_credentials(self):
        """Get real authentication credentials if available."""
        return {
            "client_id": os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
            "client_secret": os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            "token_endpoint": os.getenv("SECUREGPT_ONELOGIN_URL"),
        }

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_sync(self, real_auth_credentials):
        """Test real authentication with actual OneLogin API call (sync)."""
        auth = OneloginAuth()

        # This should make a real HTTP call to OneLogin
        token = auth.get_access_token()

        # Verify we got a real token
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # JWT tokens typically have 3 parts separated by dots
        assert token.count(".") >= 2, "Token should be a valid JWT format"

        # Verify token is stored
        assert auth.token is not None
        assert auth.token.access_token == token
        assert not auth.token.is_expired()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    async def test_real_authentication_async(self, real_auth_credentials):
        """Test real authentication with actual OneLogin API call (async)."""
        auth = OneloginAuth()

        # This should make a real HTTP call to OneLogin
        token = await auth.get_access_token_async()

        # Verify we got a real token
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # JWT tokens typically have 3 parts separated by dots
        assert token.count(".") >= 2, "Token should be a valid JWT format"

        # Verify token is stored
        assert auth.token is not None
        assert auth.token.access_token == token
        assert not auth.token.is_expired()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_token_reuse(self, real_auth_credentials):
        """Test that real tokens are properly reused when not expired."""
        auth = OneloginAuth()

        # First call should make HTTP request
        token1 = auth.get_access_token()

        # Second call should reuse the same token (no new HTTP request)
        token2 = auth.get_access_token()

        # Tokens should be identical
        assert token1 == token2
        assert auth.token.access_token == token1

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_with_custom_endpoint(self, real_auth_credentials):
        """Test real authentication with custom token endpoint."""
        custom_endpoint = real_auth_credentials.get("token_endpoint") or "https://onelogin.axa.com/as/token.oauth2"

        auth = OneloginAuth(
            token_endpoint=custom_endpoint,
            client_id=real_auth_credentials["client_id"],
            client_secret=real_auth_credentials["client_secret"],
        )

        token = auth.get_access_token()

        # Verify we got a real token
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        assert token.count(".") >= 2, "Token should be a valid JWT format"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_error_handling(self):
        """Test real authentication error handling with invalid credentials."""
        # Use invalid credentials to test error handling
        auth = OneloginAuth(
            client_id="invalid_client_id",
            client_secret="invalid_client_secret",
            token_endpoint=os.getenv("SECUREGPT_ONELOGIN_URL", "https://onelogin.axa.com/as/token.oauth2"),
        )

        # This should raise an exception due to invalid credentials
        with pytest.raises(Exception) as exc_info:
            auth.get_access_token()

        # Verify it's an authentication-related error
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in [
                "unauthorized",
                "invalid",
                "client",
                "authentication",
                "400",
                "401",
                "403",
                "error",
                "remote endpoint",
            ]
        ), f"Expected authentication error, got: {exc_info.value}"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_with_invalid_endpoint(self, real_auth_credentials):
        """Test real authentication with invalid token endpoint."""
        auth = OneloginAuth(
            token_endpoint="https://invalid.endpoint.com/token",
            client_id=real_auth_credentials["client_id"],
            client_secret=real_auth_credentials["client_secret"],
        )

        # This should raise an exception due to invalid endpoint
        with pytest.raises(Exception) as exc_info:
            auth.get_access_token()

        # Verify it's a network/endpoint-related error
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in [
                "connection",
                "network",
                "resolve",
                "timeout",
                "unreachable",
                "not found",
            ]
        ), f"Expected network error, got: {exc_info.value}"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_token_properties(self, real_auth_credentials):
        """Test properties of real authentication tokens."""
        auth = OneloginAuth()

        # Get access token but don't store in unused variable
        auth.get_access_token()

        # Verify token properties
        assert auth.token is not None
        assert hasattr(auth.token, "access_token")
        assert hasattr(auth.token, "is_expired")
        assert callable(auth.token.is_expired)

        # Test token expiration check
        is_expired = auth.token.is_expired()
        assert isinstance(is_expired, bool)
        assert not is_expired, "Freshly obtained token should not be expired"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_multiple_instances(self, real_auth_credentials):
        """Test real authentication with multiple OneloginAuth instances."""
        auth1 = OneloginAuth()
        auth2 = OneloginAuth()

        # Both should be able to authenticate independently
        token1 = auth1.get_access_token()
        token2 = auth2.get_access_token()

        # Verify both got valid tokens
        assert token1 is not None
        assert token2 is not None
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) > 0
        assert len(token2) > 0

        # Both tokens should be valid JWT format
        assert token1.count(".") >= 2, "Token1 should be a valid JWT format"
        assert token2.count(".") >= 2, "Token2 should be a valid JWT format"

        # Tokens may be different (each auth call can generate unique tokens)
        # but both should be valid and functional

    @pytest.mark.integration
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    def test_real_authentication_performance(self, real_auth_credentials):
        """Test performance of real authentication calls."""
        import time

        auth = OneloginAuth()

        # First call - should make HTTP request
        start_time = time.time()
        token1 = auth.get_access_token()
        first_call_time = time.time() - start_time

        # Second call - should reuse token
        start_time = time.time()
        token2 = auth.get_access_token()
        second_call_time = time.time() - start_time

        # Verify tokens are the same
        assert token1 == token2

        # Second call should be significantly faster (token reuse)
        assert second_call_time < first_call_time / 2, (
            f"Token reuse should be faster: {second_call_time:.3f}s vs {first_call_time:.3f}s"
        )

        # First call should complete within reasonable time
        assert first_call_time < 30.0, f"Authentication took too long: {first_call_time:.3f}s"

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not all(
            [
                os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID"),
                os.getenv("SECUREGPT_ONELOGIN_SECRET"),
            ]
        ),
        reason="Real OneLogin credentials not available in environment",
    )
    async def test_real_authentication_sync_async_consistency(self, real_auth_credentials):
        """Test that sync and async authentication methods return consistent token formats."""
        auth1 = OneloginAuth()
        auth2 = OneloginAuth()

        # Get token using sync method
        sync_token = auth1.get_access_token()

        # Get token using async method
        async_token = await auth2.get_access_token_async()

        # Both should be valid JWT tokens with same format
        assert sync_token.count(".") >= 2, "Sync token should be a valid JWT format"
        assert async_token.count(".") >= 2, "Async token should be a valid JWT format"

        # Both tokens should be valid strings
        assert isinstance(sync_token, str)
        assert isinstance(async_token, str)
        assert len(sync_token) > 0
        assert len(async_token) > 0

        # Both should be functional (not expired when freshly obtained)
        assert not auth1.token.is_expired()
        assert not auth2.token.is_expired()

        # Note: Tokens may be different due to unique JTI (JWT ID) but both should be valid
