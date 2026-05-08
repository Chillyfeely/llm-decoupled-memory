# Decoupled Working Memory for Agentic LLMs

## Project goal

Standard agentic setups (e.g. ReAct-style tool loops) grow **context bloat**: tool schemas and tool outputs accumulate in the chat history. Over many turns that raises prompt tokens, cost, and latency.

This repo compares two **LangGraph** agents on the same scripted hotel-booking transcripts:

- **Baseline** — full message history (including tool calls and tool results) in the model context.
- **Decoupled** — a **`working_memory`** dict holds structured booking facts; older turns are **compressed** (plain user text and assistant text only, no past tool traces). The **current** user turn still includes full tool-call + tool-result messages so the model can finish each step reliably.

`evaluate.py` runs both agents, compares prompt-token usage and latency, runs a paired t-test, and writes plots plus a CSV.

## Repository layout

| File | Role |
|------|------|
| `shared.py` | Mock PMS room codes, tools, and fixed conversation transcripts. |
| `baseline_agent.py` | Standard graph: tools stay in history. |
| `decoupled_agent.py` | Separate `working_memory` + compressed prior turns. |
| `evaluate.py` | Harness: invokes both agents, stats, plots, CSV. |

## Prerequisites

- **Python** 3.10+ (3.11+ recommended).
- A **Google AI Studio** API key for the Gemini API: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

The **langchain-google-genai** integration reads the key from the environment. Use either variable (both are supported):

- `GOOGLE_API_KEY` (preferred)
- `GEMINI_API_KEY`

Do not commit keys. If the key is missing, model calls fail at runtime with an API-key error.

## End-to-end setup and run

### 1. Open a terminal in the project root

```powershell
cd path\to\llm-decoupled-memory
```

### 2. Install dependencies

**Using uv (recommended):**

```powershell
uv venv
uv pip install -r requirements.txt
```

**Using pip:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set the Gemini API key

**Windows PowerShell (current session only):**

```powershell
$env:GOOGLE_API_KEY = "your-key-here"
```

**Windows Command Prompt:**

```cmd
set GOOGLE_API_KEY=your-key-here
```

**macOS / Linux (bash):**

```bash
export GOOGLE_API_KEY="your-key-here"
```

To persist on your machine, use your OS or shell’s normal mechanism (User environment variables on Windows, `~/.bashrc` / `~/.zshrc`, etc.). Restart the terminal after changing persistent variables.

### 4. Run the evaluation

**With uv:**

```powershell
uv run python evaluate.py
```

**With an activated venv:**

```powershell
python evaluate.py
```

The script logs progress to the console (including `langgraph_recursion_limit`). It needs network access to call Gemini.

### 5. Outputs (written next to the project root)

| Artifact | Description |
|----------|-------------|
| `evaluation_results.csv` | Per-turn rows: agent, prompt tokens, latency. |
| `tokens_plot.png` | Prompt tokens vs turn, baseline vs decoupled. |
| `latency_plot.png` | Latency vs turn, baseline vs decoupled. |

## Configuration notes

- **Model** is set in `baseline_agent.py` and `decoupled_agent.py` (`ChatGoogleGenerativeAI`, e.g. `gemini-3.1-flash-lite-preview`). Change there if you use another Gemini ID.
- **Graph recursion cap** is in `evaluate.py` (`LANGGRAPH_RECURSION_LIMIT`) to bound tool-loop depth per user turn.

## Troubleshooting

- **401 / API key errors** — Key not set, typo, or wrong variable. Confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the same terminal session you use to run `evaluate.py`.
- **Quota or billing** — Check Google AI Studio usage limits and project billing.
- **LangGraph recursion error** — Rare with the current decoupled prompt; you can raise `LANGGRAPH_RECURSION_LIMIT` in `evaluate.py` if a transcript needs more tool steps.
