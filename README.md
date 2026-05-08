# Decoupled Working Memory for Agentic LLMs

## Project Goal
Standard LLM conversational agents (like ReAct) suffer from "context bloat" when executing complex API actions. Tool schemas and scratchpad outputs are dumped into the conversational history. Over a multi-turn conversation, this increases prompt tokens, API costs, and latency. 

This project uses **LangGraph** to implement a "Decoupled Working Memory." We separate the standard conversational history (user-facing) from a hidden, backend state dictionary. 

## Repository Contents
* `shared.py`: Contains the mock Property Management System (PMS) schemas and the simulated transcript dataset.
* `baseline_agent.py`: A standard LangGraph implementation where tool outputs stay in the message history.
* `decoupled_agent.py`: The proposed architecture. It explicitly filters `ToolMessages` out of the chat history and maintains them in a separate `working_memory` TypedDict state.
* `evaluate.py`: The evaluation script that runs both agents across transcripts, runs the statistical Paired t-tests, and generates graphs.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt