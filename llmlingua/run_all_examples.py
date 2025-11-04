# -*- coding: utf-8 -*-
"""
Run All LLMLingua-2 Examples

This script runs all LLMLingua-2 example scripts sequentially.
Each script tests a different aspect of prompt compression.
"""

import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Make sure to set te env to disable XET transfer from HF
# and to have the certificates pointed correctly
# HF_HUB_DISABLE_XET=true
load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# List of example scripts in order
EXAMPLE_SCRIPTS = [
    "01_setup_llmlingua2.py",
    "02_indomain_meetingbank.py",
    "03_outdomain_singledoc_qa.py",
    "04_outdomain_multidoc_qa.py",
    "05_outdomain_summarization.py",
    "06_outdomain_incontext_learning.py",
]


def run_script(script_path):
    """Run a Python script and return whether it succeeded."""
    print("\n" + "=" * 80)
    print(f"Running: {script_path.name}")
    print("=" * 80 + "\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], cwd=SCRIPT_DIR, check=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_path.name}")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running {script_path.name}: {e}")
        return False


def main():
    """Run all example scripts."""
    print("=" * 80)
    print("LLMLingua-2 Examples - Running All Scripts")
    print("=" * 80)

    results = {}

    for script_name in EXAMPLE_SCRIPTS:
        script_path = SCRIPT_DIR / script_name

        if not script_path.exists():
            print(f"\n⚠️  Warning: {script_name} not found, skipping...")
            results[script_name] = "SKIPPED"
            continue

        success = run_script(script_path)
        results[script_name] = "SUCCESS" if success else "FAILED"

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for script_name, status in results.items():
        emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
        print(f"{emoji} {script_name}: {status}")

    # Return exit code based on results
    if all(status == "SUCCESS" for status in results.values()):
        print("\n🎉 All examples completed successfully!")
        return 0
    elif any(status == "FAILED" for status in results.values()):
        print("\n⚠️  Some examples failed. Check the output above for details.")
        return 1
    else:
        print("\n✅ All available examples completed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
