"""Test script to verify HTTP 429 retry logic with extracted delay."""

import re


def extract_retry_after_seconds(error_text: str) -> float | None:
    """Extract retry-after seconds from error message.

    Parses error messages like:
    "Please retry after 43 seconds."

    Args:
        error_text: The error message text

    Returns:
        Number of seconds to wait, or None if not found
    """
    # Pattern to match "retry after X seconds" or "Please retry after X seconds"
    pattern = r"retry after (\d+) seconds?"
    match = re.search(pattern, error_text, re.IGNORECASE)

    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None

    return None


def test_extract_retry_after_seconds():
    """Test the extraction function with various error messages."""
    # Test case 1: Full error message from the user
    error_msg_1 = """Error invoking secureGPT completions endpoint: HTTP 429: {"error":{"code":"RateLimitReached","message": "Your requests to gpt-4-turbo-2024-04-09 for gpt-4 in Sweden Central have exceeded the token rate limit for your current OpenAI S0 pricing tier. This request was for ChatCompletions_Create under Azure OpenAI API version 2024-10-21. Please retry after 43 seconds. To increase your default rate limit, visit: https://aka.ms/oai/quotaincrease."}}"""

    result_1 = extract_retry_after_seconds(error_msg_1)
    print("Test 1: Full error message")
    print("  Expected: 43.0")
    print(f"  Got: {result_1}")
    print(f"  Status: {'✓ PASS' if result_1 == 43.0 else '✗ FAIL'}\n")

    # Test case 2: Shorter message
    error_msg_2 = "Please retry after 60 seconds."
    result_2 = extract_retry_after_seconds(error_msg_2)
    print("Test 2: Short message")
    print("  Expected: 60.0")
    print(f"  Got: {result_2}")
    print(f"  Status: {'✓ PASS' if result_2 == 60.0 else '✗ FAIL'}\n")

    # Test case 3: Case insensitive
    error_msg_3 = "RETRY AFTER 15 SECONDS"
    result_3 = extract_retry_after_seconds(error_msg_3)
    print("Test 3: Case insensitive")
    print("  Expected: 15.0")
    print(f"  Got: {result_3}")
    print(f"  Status: {'✓ PASS' if result_3 == 15.0 else '✗ FAIL'}\n")

    # Test case 4: Singular "second"
    error_msg_4 = "retry after 1 second"
    result_4 = extract_retry_after_seconds(error_msg_4)
    print("Test 4: Singular 'second'")
    print("  Expected: 1.0")
    print(f"  Got: {result_4}")
    print(f"  Status: {'✓ PASS' if result_4 == 1.0 else '✗ FAIL'}\n")

    # Test case 5: No retry information
    error_msg_5 = "Rate limit exceeded. Please try again later."
    result_5 = extract_retry_after_seconds(error_msg_5)
    print("Test 5: No retry information")
    print("  Expected: None")
    print(f"  Got: {result_5}")
    print(f"  Status: {'✓ PASS' if result_5 is None else '✗ FAIL'}\n")

    # Test case 6: Lowercase variant
    error_msg_6 = "please retry after 120 seconds."
    result_6 = extract_retry_after_seconds(error_msg_6)
    print("Test 6: Lowercase variant")
    print("  Expected: 120.0")
    print(f"  Got: {result_6}")
    print(f"  Status: {'✓ PASS' if result_6 == 120.0 else '✗ FAIL'}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing HTTP 429 Retry Delay Extraction")
    print("=" * 60 + "\n")
    test_extract_retry_after_seconds()
    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)
