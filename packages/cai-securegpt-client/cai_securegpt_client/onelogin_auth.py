import os
from typing import Optional

from dotenv import load_dotenv
from requests_oauth2client import BearerToken, OAuth2Client

load_dotenv()


class OneloginAuth:
    """Class to manage OneLogin authentication."""

    def __init__(
        self,
        token_endpoint: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """Constructor.

        Args:
            token_endpoint (str): The full one login token endpoint.
            client_id (str): The client id.
            client_secret (str): The client secret.
        """
        self.client_id = client_id if client_id else os.getenv("SECUREGPT_ONELOGIN_CLIENT_ID")
        self.client_secret = client_secret if client_secret else os.getenv("SECUREGPT_ONELOGIN_SECRET")
        self.token_endpoint = (
            token_endpoint
            if token_endpoint
            else os.getenv("SECUREGPT_ONELOGIN_URL", "https://onelogin.axa.com/as/token.oauth2")
        )
        self.token: BearerToken | None = None

    def _get_access_token(self) -> str:
        if self.token is not None and not self.token.is_expired():
            return self.token.access_token
        oauth2client = OAuth2Client(
            token_endpoint=self.token_endpoint,
            auth=(self.client_id, self.client_secret),
        )
        self.token = oauth2client.client_credentials(scope="urn:grp:chatgpt")
        jwt_token = self.token.access_token
        return jwt_token

    def get_access_token(self) -> str:
        """Requests or returns the access token synchronously.

        Returns:
            str: The access token.
        """
        return self._get_access_token()

    async def get_access_token_async(self):
        """Requests or returns the access token asynchronously.

        Returns:
            str: The access token.
        """
        # For now, just call the sync method
        # In a real implementation, this would be truly async
        return self._get_access_token()
