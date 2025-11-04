# LLMLingua-2 Examples

This directory contains a structured set of examples demonstrating the capabilities of LLMLingua-2 for prompt compression across various tasks.

## Overview

LLMLingua-2 is a task-agnostic prompt compression method that:
- Offers 3x-6x faster performance than LLMLingua
- Excels at out-of-domain compression
- Uses a BERT-level encoder trained via data distillation from GPT-4
- Maintains high quality while significantly reducing token usage

**Reference**: [LLMLingua-2 Paper](https://aclanthology.org/2024.findings-acl.57/)

## Project Structure

```
llmlingua/
├── README.md                              # This file
├── 01_setup_llmlingua2.py                # Setup and initialization
├── 02_indomain_meetingbank.py            # In-domain testing (MeetingBank)
├── 03_outdomain_singledoc_qa.py          # Single-document QA (NarrativeQA)
├── 04_outdomain_multidoc_qa.py           # Multi-document QA (TriviaQA)
├── 05_outdomain_summarization.py         # Summarization (Gov Reports)
├── 06_outdomain_incontext_learning.py    # In-context learning (GSM8K)
└── run_all_examples.py                   # Run all examples sequentially
```

## Scripts

### 01_setup_llmlingua2.py
Initializes the LLMLingua-2 compressor and SecureGPT client.

### 02_indomain_meetingbank.py
**In-Domain Testing**: Tests on MeetingBank dataset (training domain)
- **Task**: Question answering on meeting transcripts
- **Compression**: 33% rate (~2000 tokens)
- **Dataset**: MeetingBank test set

### 03_outdomain_singledoc_qa.py
**Out-of-Domain Testing**: Single-document question answering
- **Task**: Answer questions about stories/movie scripts
- **Compression**: Target 3000 tokens
- **Dataset**: NarrativeQA from LongBench

### 04_outdomain_multidoc_qa.py
**Out-of-Domain Testing**: Multi-document question answering
- **Task**: Answer trivia questions using multiple passages
- **Compression**: Target 2000 tokens with context-level filtering
- **Dataset**: TriviaQA from LongBench

### 05_outdomain_summarization.py
**Out-of-Domain Testing**: Document summarization
- **Task**: Summarize government reports
- **Compression**: Target 3000 tokens
- **Dataset**: Gov Report from LongBench

### 06_outdomain_incontext_learning.py
**Out-of-Domain Testing**: Few-shot learning for math reasoning
- **Task**: Math problem solving with chain-of-thought
- **Compression**: Target 150 tokens with digit preservation
- **Dataset**: GSM8K

## Requirements

```bash
# Install dependencies
uv add llmlingua datasets httpx 
uv add packages/cai-securegpt-client
```

## Usage

### Run Individual Examples

```bash
# Setup (optional, models are initialized in each script)
python 01_setup_llmlingua2.py

# Run specific examples
python 02_indomain_meetingbank.py
python 03_outdomain_singledoc_qa.py
python 04_outdomain_multidoc_qa.py
python 05_outdomain_summarization.py
python 06_outdomain_incontext_learning.py
```

### Run All Examples

```bash
python run_all_examples.py
```

## Key Features Demonstrated

1. **In-Domain Performance**: High accuracy on MeetingBank data
2. **Out-of-Domain Generalization**: Effective compression on unseen domains
3. **Multiple Task Types**: QA, summarization, reasoning
4. **Context-Level Filtering**: Better handling of multi-document scenarios
5. **Digit Preservation**: Maintaining numerical accuracy in math problems
6. **Token Control**: Both rate-based and target-token compression

## Notes

- Each script is self-contained with its own model initialization
- Token usage and compression ratios are printed for comparison
- Scripts use SecureGPT for LLM inference
- Original and compressed prompts are tested side-by-side

## Original Source

Adapted from: [LLMLingua GitHub Repository](https://github.com/microsoft/LLMLingua/blob/main/examples/LLMLingua2.ipynb)

