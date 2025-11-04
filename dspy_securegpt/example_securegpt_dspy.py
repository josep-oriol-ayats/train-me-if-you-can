#!/usr/bin/env python3
"""
Example: Using SecureGPT with DSPy

This example demonstrates how to use SecureGPT models with DSPy through
the LiteLLM integration.
"""

import dspy
from dspy_securegpt.config import configure_securegpt_lm



def basic_qa_example():
    """Simple question-answering example."""
    print("=== Basic Q&A Example ===\n")

    # Configure SecureGPT LM
    lm = configure_securegpt_lm(
        model_id="gpt-4-turbo-2024-04-09",
        temperature=0.7,
        max_tokens=1000
    )

    # Set as default LM
    dspy.configure(lm=lm)

    # Simple prompt
    response = lm("What are the key principles of machine learning?")
    print(f"Response: {response}")
    print()


def chain_of_thought_example():
    """Example using DSPy's ChainOfThought."""
    print("=== Chain of Thought Example ===\n")

    # Configure SecureGPT LM
    lm = configure_securegpt_lm(
        model_id="gpt-4-turbo-2024-04-09",
        temperature=0.5
    )

    dspy.configure(lm=lm)

    # Define a simple signature
    class QA(dspy.Signature):
        """Answer questions with reasoning."""
        question = dspy.InputField()
        answer = dspy.OutputField()

    # Create a ChainOfThought module
    cot = dspy.ChainOfThought(QA)

    # Ask a question that benefits from reasoning
    question = "If a train travels 120 km in 2 hours, what is its average speed?"
    response = cot(question=question)

    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print()


def multi_hop_example():
    """Example with multi-hop reasoning."""
    print("=== Multi-Hop Reasoning Example ===\n")

    # Configure SecureGPT LM
    lm = configure_securegpt_lm(
        model_id="gpt-4-turbo-2024-04-09",
        temperature=0.3
    )

    dspy.configure(lm=lm)

    # Define a signature for complex reasoning
    class ComplexReasoning(dspy.Signature):
        """Answer complex questions requiring multiple reasoning steps."""
        context = dspy.InputField(desc="background information")
        question = dspy.InputField()
        reasoning_steps = dspy.OutputField(desc="step by step reasoning")
        final_answer = dspy.OutputField()

    # Create the module
    reasoner = dspy.ChainOfThought(ComplexReasoning)

    # Example with context
    context = """
    Company A has 100 employees and grows by 20% each year.
    Company B has 150 employees and grows by 10% each year.
    """

    question = "In how many years will Company A have more employees than Company B?"

    response = reasoner(context=context, question=question)

    print(f"Context: {context.strip()}")
    print(f"Question: {question}")
    print(f"Reasoning: {response.reasoning_steps}")
    print(f"Answer: {response.final_answer}")
    print()


def predict_example():
    """Example using DSPy's Predict for structured outputs."""
    print("=== Structured Output Example ===\n")

    # Configure SecureGPT LM
    lm = configure_securegpt_lm(
        model_id="gpt-4-turbo-2024-04-09",
        temperature=0.4
    )

    dspy.configure(lm=lm)

    # Define a signature for sentiment analysis
    class SentimentAnalysis(dspy.Signature):
        """Analyze the sentiment of text."""
        text = dspy.InputField()
        sentiment = dspy.OutputField(desc="positive, negative, or neutral")
        confidence = dspy.OutputField(desc="confidence score from 0 to 1")
        reasoning = dspy.OutputField(desc="brief explanation")

    # Create predictor
    classifier = dspy.Predict(SentimentAnalysis)

    # Analyze some text
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Complete waste of money.",
        "It's okay, nothing special but does the job."
    ]

    for text in texts:
        result = classifier(text=text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print()


def comparison_example():
    """Compare different temperatures."""
    print("=== Temperature Comparison Example ===\n")

    question = "Write a creative one-line story about a robot."

    for temp in [0.0, 0.7, 1.5]:
        lm = configure_securegpt_lm(
            model_id="gpt-4-turbo-2024-04-09",
            temperature=temp,
            max_tokens=100
        )
        dspy.configure(lm=lm)

        response = lm(question)
        print(f"Temperature {temp}: {response}")
        print()


def main():
    """Run all examples."""
    print("SecureGPT + DSPy Integration Examples")
    print("=" * 50)
    print()

    try:
        # Run examples
        basic_qa_example()
        # chain_of_thought_example()
        # multi_hop_example()
        # predict_example()
        # comparison_example()

        print("=" * 50)
        print("Basic example completed successfully!")
        print("Note: Complex examples (Chain of Thought, etc.) may require")
        print("additional adapter configuration for DSPy 2.x")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nMake sure you have:")
        print("1. SecureGPT credentials configured (OneLogin or access token)")
        print("2. Environment variables set if needed (SECUREGPT_URL, etc.)")
        print("3. Network access to SecureGPT API")
        raise


if __name__ == "__main__":
    main()

