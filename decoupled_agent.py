"""
Decoupled LangGraph agent: external working memory with filtered chat history.
"""

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from shared import PMS_ROOM_CODES, execute_booking, update_working_memory

logger = logging.getLogger(__name__)


class DecoupledState(TypedDict):
    """Graph state: trimmed messages plus an explicit working-memory dictionary."""

    messages: Annotated[list, add_messages]
    working_memory: dict


tools = [update_working_memory, execute_booking]
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0).bind_tools(tools)


def _index_of_last_human_message(messages: list) -> int:
    """Return the index of the most recent HumanMessage, or -1 if none."""

    last_index = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_index = idx
    return last_index


def _compressed_prefix_before_index(messages: list, past_end_index: int) -> list:
    """Return user-facing turns only (no tool calls or tool results) for messages before past_end_index."""

    out = []
    for msg in messages[:past_end_index]:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue
        out.append(msg)
    return out


def _memory_after_update_args(base_memory: dict, args: dict) -> dict:
    """Return a copy of base_memory after applying update_working_memory fields from args."""

    result = dict(base_memory)
    if "room_type" in args:
        room_type_normalized = args["room_type"].lower()
        result["room_type"] = room_type_normalized
        result["room_code"] = PMS_ROOM_CODES.get(room_type_normalized)
    if "guests" in args:
        result["guests"] = args["guests"]
    if "check_in_date" in args:
        result["check_in_date"] = args["check_in_date"]
    return result


def chatbot_node(state: DecoupledState):
    """Call the model with system + memory snapshot and a filtered message list."""

    working_memory = state.get("working_memory", {})
    logger.debug(
        "decoupled chatbot_node enter raw_messages=%s working_memory=%s",
        len(state["messages"]),
        working_memory,
    )
    sys_msg = SystemMessage(
        content=f"""
    You are a hotel booking assistant with a decoupled working memory.
    CURRENT WORKING MEMORY: {json.dumps(working_memory)}
    
    Available Room Codes to save to memory: {json.dumps(PMS_ROOM_CODES)}
    
    Use 'update_working_memory' only for NEW or CHANGED facts. If CURRENT WORKING MEMORY already lists the same room_type, room_code, guests, or check_in_date, do not call that tool again for that fact.
    Respond with normal text when no tool is needed. Do not book until the user asks to finalize.
    When ready, use 'execute_booking'.
    """
    )

    last_human_index = _index_of_last_human_message(state["messages"])
    if last_human_index < 0:
        compressed_prefix: list = []
        current_user_turn_messages = state["messages"]
    else:
        compressed_prefix = _compressed_prefix_before_index(state["messages"], last_human_index)
        current_user_turn_messages = state["messages"][last_human_index:]

    prompt_messages = [sys_msg] + compressed_prefix + current_user_turn_messages
    logger.debug(
        "decoupled chatbot_node prefix=%s current_turn=%s prompt_tail_types=%s",
        len(compressed_prefix),
        len(current_user_turn_messages),
        [type(message_item).__name__ for message_item in prompt_messages[-8:]],
    )
    response = llm.invoke(prompt_messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info(
        "decoupled chatbot_node exit tool_call_count=%s response_type=%s",
        len(tool_calls),
        type(response).__name__,
    )
    if tool_calls:
        logger.debug("decoupled chatbot_node tool_calls=%s", tool_calls)
    return {"messages": [response]}


def tool_node(state: DecoupledState):
    """Apply tool side effects, merge memory updates, and return tool messages."""

    last_msg = state["messages"][-1]
    new_memory = state.get("working_memory", {}).copy()
    tool_calls = last_msg.tool_calls
    logger.info("decoupled tool_node batch_size=%s memory_before=%s", len(tool_calls), new_memory)
    tool_responses = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args")
        logger.info("decoupled tool_node tool=%s args=%s", tool_name, tool_args)
        if tool_name == "update_working_memory":
            args = tool_call["args"]
            memory_candidate = _memory_after_update_args(new_memory, args)
            if memory_candidate == new_memory:
                logger.info("decoupled tool_node redundant update_working_memory skipped")
                tool_responses.append(
                    ToolMessage(
                        content="No change: working memory already matches. Answer the user with plain text only.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            new_memory = memory_candidate
            tool_responses.append(ToolMessage(content="Memory updated silently.", tool_call_id=tool_call["id"]))

        elif tool_name == "execute_booking":
            msg = execute_booking.invoke(tool_call["args"])
            tool_responses.append(ToolMessage(content=msg, tool_call_id=tool_call["id"]))
        else:
            logger.warning("decoupled tool_node unknown_tool=%s", tool_name)

    logger.debug("decoupled tool_node memory_after=%s", new_memory)
    return {"messages": tool_responses, "working_memory": new_memory}


def should_continue(state: DecoupledState):
    """Route to tools when the model issued tool calls; otherwise finish the step."""

    last = state["messages"][-1]
    has_tools = bool(getattr(last, "tool_calls", None))
    if has_tools:
        logger.debug(
            "decoupled should_continue -> tools (n_calls=%s)",
            len(last.tool_calls),
        )
        return "tools"
    logger.debug("decoupled should_continue -> END")
    return END


workflow = StateGraph(DecoupledState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", should_continue)
workflow.add_edge("tools", "chatbot")
decoupled_app = workflow.compile()
