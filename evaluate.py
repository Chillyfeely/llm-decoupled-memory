"""
Compare baseline and decoupled LangGraph agents on fixed transcripts.

Logs timings, token estimates per turn, and post-run statistics.
"""

import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from scipy import stats

from baseline_agent import baseline_app
from decoupled_agent import decoupled_app
from shared import TRANSCRIPTS


LANGGRAPH_RECURSION_LIMIT = 40

LANGGRAPH_INVOKE_CONFIG = {"recursion_limit": LANGGRAPH_RECURSION_LIMIT}


def configure_evaluation_logging(log_level: int = logging.DEBUG) -> None:
    """Attach a stderr handler and tune third-party log noise for evaluation runs."""

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        root.setLevel(log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("langchain_google_genai").setLevel(logging.INFO)


logger = logging.getLogger(__name__)


def get_tokens_from_state(state):
    """Extract reported input token usage from the last Gemini message, if present."""

    last_msg = state["messages"][-1]
    usage = getattr(last_msg, "usage_metadata", None)
    if usage is not None:
        tokens = usage.get("input_tokens", 0)
        logger.debug(
            "get_tokens_from_state: type=%s input_tokens=%s full_usage=%s",
            type(last_msg).__name__,
            tokens,
            usage,
        )
        return tokens
    logger.debug("get_tokens_from_state: no usage_metadata on %s", type(last_msg).__name__)
    return 0


def _preview_text(text: str, max_chars: int = 160) -> str:
    """Return a single-line preview of user or model text for logs."""

    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."


def run_evaluation() -> None:
    """Run all transcripts through both agents and emit metrics, plots, and CSV."""

    results = []
    total_turns = sum(len(transcript_rows) for transcript_rows in TRANSCRIPTS)

    logger.info(
        "Starting evaluation: model_path=GEMINI conversations=%s total_turns=%s langgraph_recursion_limit=%s",
        len(TRANSCRIPTS),
        total_turns,
        LANGGRAPH_RECURSION_LIMIT,
    )

    for conv_idx, transcript in enumerate(TRANSCRIPTS):
        logger.info(
            "Conversation %s/%s: turns=%s",
            conv_idx + 1,
            len(TRANSCRIPTS),
            len(transcript),
        )

        b_state = {"messages": []}
        d_state = {"messages": [], "working_memory": {}}

        for turn_idx, user_input in enumerate(transcript):
            logger.info(
                "Conv %s turn %s/%s user_preview=%r",
                conv_idx + 1,
                turn_idx + 1,
                len(transcript),
                _preview_text(user_input),
            )

            b_state["messages"].append(HumanMessage(content=user_input))
            logger.debug(
                "Baseline invoke start: conv=%s turn=%s prior_human_msgs=%s",
                conv_idx,
                turn_idx + 1,
                sum(1 for message_item in b_state["messages"] if isinstance(message_item, HumanMessage)),
            )
            start_time = time.perf_counter()
            try:
                b_state = baseline_app.invoke(b_state, LANGGRAPH_INVOKE_CONFIG)
            except GraphRecursionError:
                logger.error(
                    "GraphRecursionError baseline conv=%s turn=%s recursion_limit=%s",
                    conv_idx + 1,
                    turn_idx + 1,
                    LANGGRAPH_RECURSION_LIMIT,
                )
                raise
            b_latency = time.perf_counter() - start_time
            b_tokens = get_tokens_from_state(b_state)
            logger.info(
                "Baseline done conv=%s turn=%s latency_s=%.3f prompt_tokens=%s",
                conv_idx + 1,
                turn_idx + 1,
                b_latency,
                b_tokens,
            )

            results.append(
                {
                    "conversation": conv_idx,
                    "turn": turn_idx + 1,
                    "agent": "Baseline",
                    "prompt_tokens": b_tokens,
                    "latency": b_latency,
                }
            )

            d_state["messages"].append(HumanMessage(content=user_input))
            logger.debug(
                "Decoupled invoke start: conv=%s turn=%s working_memory_keys=%s",
                conv_idx,
                turn_idx + 1,
                sorted(d_state.get("working_memory", {}).keys()),
            )
            start_time = time.perf_counter()
            try:
                d_state = decoupled_app.invoke(d_state, LANGGRAPH_INVOKE_CONFIG)
            except GraphRecursionError:
                logger.error(
                    "GraphRecursionError decoupled conv=%s turn=%s recursion_limit=%s",
                    conv_idx + 1,
                    turn_idx + 1,
                    LANGGRAPH_RECURSION_LIMIT,
                )
                raise
            d_latency = time.perf_counter() - start_time
            d_tokens = get_tokens_from_state(d_state)
            logger.info(
                "Decoupled done conv=%s turn=%s latency_s=%.3f prompt_tokens=%s working_memory=%s",
                conv_idx + 1,
                turn_idx + 1,
                d_latency,
                d_tokens,
                d_state.get("working_memory", {}),
            )

            results.append(
                {
                    "conversation": conv_idx,
                    "turn": turn_idx + 1,
                    "agent": "Decoupled",
                    "prompt_tokens": d_tokens,
                    "latency": d_latency,
                }
            )

    df = pd.DataFrame(results)
    logger.debug("Results dataframe shape=%s dtypes=%s", df.shape, df.dtypes.to_dict())
    logger.debug("Results head:\n%s", df.head(10).to_string())

    baseline_tokens = df[df["agent"] == "Baseline"]["prompt_tokens"].values
    decoupled_tokens = df[df["agent"] == "Decoupled"]["prompt_tokens"].values

    t_stat, p_val = stats.ttest_rel(baseline_tokens, decoupled_tokens)

    logger.info("=" * 40)
    logger.info("STATISTICAL RESULTS (paired t-test, prompt_tokens)")
    logger.info("Average Baseline Prompt Tokens: %.2f", baseline_tokens.mean())
    logger.info("Average Decoupled Prompt Tokens: %.2f", decoupled_tokens.mean())
    logger.info("T-statistic: %.4f, P-value: %.4e", t_stat, p_val)
    if p_val < 0.05:
        logger.info("Result: SIGNIFICANT difference (alpha=0.05).")
    else:
        logger.info("Result: No significant difference (alpha=0.05).")
    logger.info("=" * 40)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="turn", y="prompt_tokens", hue="agent", marker="o")
    plt.title("Prompt Tokens per Conversation Turn (gemini-3.1-flash-lite-preview)")
    plt.xlabel("Turn Number")
    plt.ylabel("Prompt Tokens")
    plt.xticks(range(1, df["turn"].max() + 1))
    plt.savefig("tokens_plot.png", dpi=300, bbox_inches="tight")
    logger.info("Wrote plot tokens_plot.png")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="turn", y="latency", hue="agent", marker="s")
    plt.title("Response Latency per Conversation Turn")
    plt.xlabel("Turn Number")
    plt.ylabel("Latency (seconds)")
    plt.xticks(range(1, df["turn"].max() + 1))
    plt.savefig("latency_plot.png", dpi=300, bbox_inches="tight")
    logger.info("Wrote plot latency_plot.png")

    df.to_csv("evaluation_results.csv", index=False)
    logger.info("Wrote evaluation_results.csv rows=%s", len(df))


if __name__ == "__main__":
    configure_evaluation_logging()
    run_evaluation()
