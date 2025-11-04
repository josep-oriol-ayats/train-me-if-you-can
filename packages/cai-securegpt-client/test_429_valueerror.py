"""Test script to demonstrate HTTP 429 retry logic catching ValueError."""


def simulate_429_error_scenario():
    """Simulate what happens when a 429 error is raised as ValueError."""
    # Simulate the error message that would be raised
    error_msg = 'Error invoking secureGPT completions endpoint: HTTP 429: {"error":{"code":"RateLimitReached","message": "Your requests to gpt-4-turbo-2024-04-09 for gpt-4 in Sweden Central have exceeded the token rate limit for your current OpenAI S0 pricing tier. This request was for ChatCompletions_Create under Azure OpenAI API version 2024-10-21. Please retry after 43 seconds. To increase your default rate limit, visit: https://aka.ms/oai/quotaincrease."}}'

    print("Simulating 429 error scenario...")
    print(f"\nError message: {error_msg[:100]}...")

    # Test the helper functions
    import re

    def _is_429_error(error_message: str) -> bool:
        """Check if error message contains HTTP 429 status."""
        return "HTTP 429" in error_message or "429" in error_message

    def _extract_retry_after_seconds(error_text: str) -> float | None:
        """Extract retry-after seconds from error message."""
        pattern = r"retry after (\d+) seconds?"
        match = re.search(pattern, error_text, re.IGNORECASE)

        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return None

        return None

    # Test detection
    is_429 = _is_429_error(error_msg)
    print(f"\n✓ Is 429 error detected: {is_429}")

    # Test extraction
    retry_after = _extract_retry_after_seconds(error_msg)
    print(f"✓ Extracted retry delay: {retry_after} seconds")

    # Simulate retry logic
    if is_429 and retry_after:
        print(f"\n✓ Would wait {retry_after} seconds before retry")
        print("✓ Then retry the request")

    return is_429, retry_after


def demonstrate_flow():
    """Demonstrate the complete flow of catching ValueError and retrying."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: ValueError Retry Logic")
    print("=" * 70)

    print("\nScenario: _generate() catches ValueError from completion_with_retry()")
    print("-" * 70)

    max_retries = 2

    for retry_attempt in range(max_retries + 1):
        print(f"\nAttempt {retry_attempt + 1}:")

        try:
            # Simulate the ValueError being raised
            if retry_attempt == 0:
                error_msg = 'Error invoking secureGPT completions endpoint: HTTP 429: {"error":{"message": "Please retry after 5 seconds."}}'
                print(f"  ✗ ValueError raised: {error_msg[:80]}...")
                raise ValueError(error_msg)
            else:
                print("  ✓ Request succeeded!")
                return "Success!"

        except ValueError as e:
            error_msg = str(e)

            # Check if this is a 429 error
            is_429 = "HTTP 429" in error_msg or "429" in error_msg

            if is_429 and retry_attempt < max_retries:
                # Extract retry delay
                import re

                pattern = r"retry after (\d+) seconds?"
                match = re.search(pattern, error_msg, re.IGNORECASE)

                if match:
                    retry_after = float(match.group(1))
                    print(f"  → 429 detected, extracted delay: {retry_after}s")
                    print(f"  → Waiting {retry_after} seconds...")
                    # In real scenario: time.sleep(retry_after)
                    print("  → Retrying...")
                    continue

            # Re-raise if not 429, no retries left, or no delay found
            print("  ✗ Re-raising exception (no more retries or not 429)")
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("HTTP 429 ValueError Retry Logic Test")
    print("=" * 70)

    # Test 1: Error detection and extraction
    is_429, retry_after = simulate_429_error_scenario()

    assert is_429 == True, "Should detect 429 error"
    assert retry_after == 43.0, f"Should extract 43 seconds, got {retry_after}"

    print("\n✓ All assertions passed!")

    # Test 2: Demonstrate the retry flow
    result = demonstrate_flow()

    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)
