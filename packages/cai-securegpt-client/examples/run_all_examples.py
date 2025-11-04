#!/usr/bin/env python3
"""
Run all example scripts in the examples/ folder.

This script executes all the example scripts to demonstrate the capabilities
of the SecureGPTChat client library.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_example_script(script_path: str) -> bool:
    """
    Run an example script and return True if successful.

    Args:
        script_path: Path to the example script

    Returns:
        True if the script ran successfully, False otherwise
    """
    script_name = os.path.basename(script_path)

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}")

    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            timeout=120,  # 2 minute timeout per script
        )

        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def get_example_scripts() -> list[str]:
    """
    Get all example scripts in the examples directory.

    Returns:
        List of paths to example scripts
    """
    examples_dir = Path(__file__).parent
    scripts = []

    # Get all Python files in the examples directory
    for file_path in examples_dir.glob("*.py"):
        # Skip this script itself
        if file_path.name != "run_all_examples.py":
            scripts.append(str(file_path))

    # Sort scripts for consistent execution order
    scripts.sort()
    return scripts


def main():
    """Run all example scripts."""
    print("SecureGPTChat Examples Runner")
    print("=" * 60)
    print("This script will run all example scripts in the examples/ folder.")
    print("Each script demonstrates different capabilities of the SecureGPTChat client.")
    print("\nNote: Some examples may show expected errors if authentication is not configured.")

    # Get all example scripts
    example_scripts = get_example_scripts()

    if not example_scripts:
        print("No example scripts found in the examples directory.")
        return

    print(f"\nFound {len(example_scripts)} example scripts:")
    for script in example_scripts:
        print(f"  - {os.path.basename(script)}")

    # Run each script
    successful_runs = 0
    failed_runs = 0

    for script_path in example_scripts:
        if run_example_script(script_path):
            successful_runs += 1
        else:
            failed_runs += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total scripts: {len(example_scripts)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")

    if failed_runs > 0:
        print("\nNote: Some failures are expected if:")
        print("- Authentication credentials are not configured")
        print("- Network connectivity issues")
        print("- Service is in demo/development mode")

    print(f"\n{'=' * 60}")
    print("All examples completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
