#!/usr/bin/env python3
"""
Simple test to verify SecureGPT + DSPy integration.

Run this to check if the integration is working correctly.
"""

import sys


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import dspy as dspy_pkg
        print("✓ dspy imported")

        import litellm
        print("✓ litellm imported")

        from cai_securegpt_client import SecureGPTChat
        print("✓ SecureGPTChat imported")

        # Import local dspy_securegpt config
        sys.path.insert(0, '/Users/b514xo/Projects/Train-me-if-you-can')
        from dspy_securegpt.config import configure_securegpt_lm, get_default_securegpt_lm
        print("✓ dspy_securegpt.config imported")

        import dspy_securegpt.securegpt_litellm
        print("✓ dspy_securegpt.securegpt_litellm imported")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_handler_registration():
    """Test that the custom handler is registered with LiteLLM."""
    print("\nTesting handler registration...")
    try:
        import litellm
        import dspy_securegpt.securegpt_litellm

        # Check if handler is registered
        if hasattr(litellm, 'custom_provider_map') and litellm.custom_provider_map:
            providers = [p.get('provider') for p in litellm.custom_provider_map if isinstance(p, dict)]
            if 'securegpt' in providers:
                print("✓ SecureGPT handler registered with LiteLLM")
                return True
            else:
                print(f"✗ SecureGPT not found in providers: {providers}")
                return False
        else:
            print("✗ custom_provider_map not found or empty")
            return False
    except Exception as e:
        print(f"✗ Registration check failed: {e}")
        return False


def test_lm_creation():
    """Test creating an LM instance."""
    print("\nTesting LM creation...")
    try:
        sys.path.insert(0, '/Users/b514xo/Projects/Train-me-if-you-can')
        import dspy as dspy_pkg
        from dspy_securegpt.config import configure_securegpt_lm
        import dspy_securegpt.securegpt_litellm

        # Create LM instance
        lm = configure_securegpt_lm(
            model_id="gpt-4-turbo-2024-04-09",
            temperature=0.7,
            max_tokens=100
        )

        print(f"✓ LM created: {type(lm)}")
        print(f"  Model: {lm.model}")
        print(f"  Temperature: {lm.kwargs.get('temperature')}")
        print(f"  Max tokens: {lm.kwargs.get('max_tokens')}")

        return True
    except Exception as e:
        print(f"✗ LM creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_string_format():
    """Test that model string is formatted correctly."""
    print("\nTesting model string format...")
    try:
        sys.path.insert(0, '/Users/b514xo/Projects/Train-me-if-you-can')
        from dspy_securegpt.config import configure_securegpt_lm
        import dspy_securegpt.securegpt_litellm

        lm = configure_securegpt_lm(model_id="gpt-4-turbo-2024-04-09")

        expected_prefix = "securegpt/"
        if lm.model.startswith(expected_prefix):
            print(f"✓ Model string format correct: {lm.model}")
            return True
        else:
            print(f"✗ Model string format incorrect: {lm.model}")
            print(f"  Expected to start with: {expected_prefix}")
            return False
    except Exception as e:
        print(f"✗ Model string test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SecureGPT + DSPy Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Handler Registration", test_handler_registration),
        ("LM Creation", test_lm_creation),
        ("Model String Format", test_model_string_format),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' raised exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
        print("\nNext steps:")
        print("1. Set up authentication (OneLogin or access token)")
        print("2. Run the example: python dspy_securegpt/example_securegpt_dspy.py")
        print("3. Try using SecureGPT with your DSPy modules")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

