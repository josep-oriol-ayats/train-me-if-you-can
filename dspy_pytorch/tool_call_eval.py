# Code for the blog post
# Optimizing Tool Selection for LLM Workflows: Differentiable Programming with PyTorch and DSPy

# How local, learnable routers can reduce token overhead, lower costs, and bring structure back to agentic workflows.

# https://viksit.substack.com/p/optimizing-tool-selection-for-llm
# Ping @viksit on X with feedback/questions

# ----------------------------------------------------
# MODIFIED TO USE Azure OpenAI
#
# Requirements:
# - Set AZURE_API_KEY environment variable to your Azure OpenAI API key
# - Set AZURE_API_BASE environment variable to your Azure OpenAI endpoint
# - Set AZURE_API_VERSION (optional, defaults to 2023-12-01-preview)
# - Set AZURE_DEPLOYMENT_NAME (optional, defaults to gpt-4-turbo)
#
# Example:
#   export AZURE_API_KEY="your-api-key"
#   export AZURE_API_BASE="https://your-resource.openai.azure.com"
#   export AZURE_API_VERSION="2023-12-01-preview"
#   export AZURE_DEPLOYMENT_NAME="gpt-4-turbo"
#   python dspy_pytorch/tool_call_eval.py
# ----------------------------------------------------

# tool controller
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import statistics
import time
import dspy
from random import sample
from tabulate import tabulate
import sys
import os

# Configure Azure OpenAI
try:
    print("Configuring Azure OpenAI...")

    # Get Azure OpenAI credentials from environment variables
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-turbo")

    if not azure_api_key or not azure_api_base:
        raise ValueError("AZURE_API_KEY and AZURE_API_BASE environment variables must be set")

    # Configure DSPy with Azure OpenAI
    lm = dspy.LM(
        f"azure/{azure_deployment}",
        api_key=azure_api_key,
        api_base=azure_api_base,
        api_version=azure_api_version,
        temperature=0.7,
        max_tokens=4000
    )

    # Initialize history tracking for cost analysis
    if not hasattr(lm, 'history'):
        lm.history = []

    # Configure DSPy to use Azure OpenAI
    dspy.configure(lm=lm)
    print("Azure OpenAI configured successfully!")

except Exception as e:
    print(f"Error configuring Azure OpenAI: {e}")
    print("\nPlease ensure:")
    print("1. AZURE_API_KEY environment variable is set")
    print("2. AZURE_API_BASE environment variable is set (e.g., https://your-resource.openai.azure.com)")
    print("3. AZURE_API_VERSION environment variable is set (optional, defaults to 2023-12-01-preview)")
    print("4. AZURE_DEPLOYMENT_NAME environment variable is set (optional, defaults to gpt-4-turbo)")
    sys.exit(1)


## --- simple RNN for tool controller --
## can be a more sophisticated encoder model too!

class ToolController(nn.Module):
    def __init__(self, vocab: dict, dim: int = 64):
        super().__init__()
        self.vocab, self.emb = vocab, nn.Embedding(256, dim)
        self.rnn, self.lin = nn.GRU(dim, dim, batch_first=True), nn.Linear(dim, 2)

    def _tok(self, txt):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in txt.lower().split()]
        return torch.tensor([ids])

    def forward(self, txt):
        x, _ = self.rnn(self.emb(self._tok(txt)))
        return F.gumbel_softmax(self.lin(x[:, -1]), hard=False)  # probs [2]


## --- train via a synthetic dataset -----

NUM_SYN = 400  # keep tiny so cell runs fast
search_queries = ["who is ceo of dropbox",
                  "population of japan",
                  "capital of france",
                  "define large language model"]
calc_queries = ["2 + 2", "15 * 7", "sqrt(81)", "log(100, 10)"]

dataset = [(q, 0) for q in search_queries for _ in range(NUM_SYN // 8)] + \
          [(q, 1) for q in calc_queries for _ in range(NUM_SYN // 8)]
random.shuffle(dataset)

# ------------------ 3. Train controller ------------------
vocab, net = {}, ToolController(vocab={})
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
loss_fn = nn.NLLLoss()

for epoch in range(4):
    tot = 0
    for q, label in dataset:
        probs = net(q)  # tensor [1,2]
        loss = loss_fn(torch.log(probs + 1e-9), torch.tensor([label]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    print(f"epoch {epoch} avg-loss {tot / len(dataset):.4f}")


## exercise the code above
def route(q: str):
    probs = net(q)
    tool = "SEARCH" if torch.argmax(probs).item() == 0 else "CALCULATE"
    return tool, probs.detach().numpy().round(3).tolist()


tests = ["who is ceo of dropbox",
         "2 + 2",
         "define transformers architecture",
         "sqrt(256)"]

for t in tests:
    tool, p = route(t)
    print(f"{t:<40} → {tool:10}  probs={p}")


## Add on the ToolController to DSPy


# Define DSPy tools
class SearchTool(dspy.Module):
    def forward(self, query: str) -> dspy.Prediction:
        return dspy.Prediction(result=f"SEARCH")


class CalcTool(dspy.Module):
    def forward(self, query: str) -> dspy.Prediction:
        return dspy.Prediction(result=f"CALCULATE")


# Router module that uses the controller
class DiffRouter(dspy.Module):
    def __init__(self, controller, tools: dict[str, dspy.Module]):
        super().__init__()
        self.controller = controller
        self.tools = tools
        self.tool_keys = list(tools.keys())

    def forward(self, query: str) -> dspy.Prediction:
        with torch.no_grad():
            probs = self.controller(query)
            selected = self.tool_keys[int(probs.argmax())]
        return self.tools[selected](query=query)


# Instantiate and run
vocab = {}
tools = {
    "search": SearchTool(),
    "calc": CalcTool()
}
router = DiffRouter(net, tools)

tests = ["who is ceo of dropbox",
         "2 + 2",
         "define transformers architecture",
         "sqrt(256)"]

for query in tests:
    result = router(query)
    print("Selected tool output:", result.result)

device = torch.device("mps" if torch.mps.is_available() else "cpu")


class RNNRouter(nn.Module):
    def __init__(self, vocab, dim=64):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(4096, dim, padding_idx=0)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.lin = nn.Linear(dim, 2)

    def _tok(self, txt):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1)
               for w in txt.lower().split()]
        return torch.tensor([ids], device=device)

    def forward(self, txt):
        x, _ = self.rnn(self.emb(self._tok(txt)));
        return self.lin(x[:, -1])


vocab, rnn = {}, RNNRouter(vocab).to(device)
opt, loss_fn = torch.optim.Adam(rnn.parameters(), 1e-3), nn.CrossEntropyLoss()

# Create sample training and test data for refund classification
refund_tickets = [
    "I want my money back, product is defective",
    "Please refund my order, it arrived broken",
    "Requesting a refund for the damaged item",
    "The product doesn't work, I need a refund",
    "Can I get a refund? This is not what I ordered",
]
no_refund_tickets = [
    "How do I track my order?",
    "What is your return policy?",
    "Can you help me with setup?",
    "When will my item ship?",
    "I have a question about features",
]

# Create training dataframe
train_data = []
for ticket in refund_tickets * 20:  # Repeat for more training data
    train_data.append({"ticket_text": ticket, "label": "REFUND"})
for ticket in no_refund_tickets * 20:
    train_data.append({"ticket_text": ticket, "label": "NO_REFUND"})
df_train = pd.DataFrame(train_data)

# Create test dataframe
test_data = []
for ticket in refund_tickets * 5:
    test_data.append({"ticket_text": ticket, "label": "REFUND"})
for ticket in no_refund_tickets * 5:
    test_data.append({"ticket_text": ticket, "label": "NO_REFUND"})
df_test = pd.DataFrame(test_data)

for _ in range(3):
    for row in df_train.itertuples():
        y = torch.tensor([0] if row.label == "REFUND" else [1], device=device)
        loss = loss_fn(rnn(row.ticket_text), y)
        opt.zero_grad();
        loss.backward();
        opt.step()


# --- sanity check --------------------------------------------------

def softmax_conf(logits):
    probs = F.softmax(logits, dim=-1)[0]
    return float(probs.max()), "REFUND" if probs.argmax() == 0 else "NO_REFUND"


test_snip = sample(list(df_test.itertuples()), k=10)  # 10 random tickets
hits = 0

for row in test_snip:
    with torch.no_grad():
        conf, pred = softmax_conf(rnn(row.ticket_text))
    correct = "✓" if pred == row.label else "✗"
    hits += (pred == row.label)
    print(f"{correct}  {pred:<8}  conf={conf:.2f} | {row.ticket_text[:60]}...")

print(f"\nMini-accuracy: {hits}/{len(test_snip)} = {hits / len(test_snip):.1%}")


# dspy routing and planning
class PlanSig(dspy.Signature):
    """Return exactly REFUND or NO_REFUND."""
    ticket: str = dspy.InputField()
    label: str = dspy.OutputField(desc="REFUND or NO_REFUND")


class ReplySig(dspy.Signature):
    """Write a one-sentence customer-support reply."""
    ticket: str = dspy.InputField()
    outcome: str = dspy.InputField(desc="REFUND or NO_REFUND")
    reply: str = dspy.OutputField()


class GPTRouter(dspy.Module):
    def forward(self, ticket):
        label = dspy.Predict(PlanSig)(ticket=ticket).label.strip().upper()
        if "REFUND" not in label: label = "NO_REFUND"
        reply = dspy.Predict(ReplySig)(ticket=ticket, outcome=label).reply
        return dspy.Prediction(label=label, reply=reply)


class RNNPlusGPT(dspy.Module):
    def forward(self, ticket):
        label = "REFUND" if rnn(ticket).argmax().item() == 0 else "NO_REFUND"
        reply = dspy.Predict(ReplySig)(ticket=ticket, outcome=label).reply
        return dspy.Prediction(label=label, reply=reply)


devset = [
    dspy.Example(ticket=row.ticket_text, label=row.label).with_inputs("ticket")
    for row in df_test.itertuples()
]


# accuracy eval
def accuracy_metric(ex, pred, _=None): return float(ex.label == pred.label)


eval_fn = dspy.Evaluate(metric=accuracy_metric, devset=devset, display_progress=True)

print("GPT-only agent accuracy:", eval_fn(GPTRouter()))
print("RNN + GPT agent accuracy:", eval_fn(RNNPlusGPT()))


def eval_cost(agent, name):
    # clear LM call history if it exists
    if hasattr(lm, 'history'):
        lm.history = []

    # accuracy + latency
    times, good = [], 0
    for row in df_test.itertuples():
        t0 = time.time()
        pred = agent(ticket=row.ticket_text).label
        times.append(time.time() - t0)
        good += (pred == row.label)
    acc = good / len(df_test)
    lat = statistics.median(times) * 1e3

    # $$$ from LiteLLM cost tracking (if available)
    cost = 0.0
    if hasattr(lm, 'history') and lm.history:
        cost = sum(x.get("cost", 0) for x in lm.history if isinstance(x, dict))

    return dict(router=name, acc=acc, ms=lat, usd=cost)


gpt_stats = eval_cost(GPTRouter(), "GPT-only")
rnn_stats = eval_cost(RNNPlusGPT(), "RNN + GPT")

print(
    tabulate(
        [list(gpt_stats.values()), list(rnn_stats.values())],
        headers=list(gpt_stats.keys()),
        floatfmt=".3f",
    )
)
