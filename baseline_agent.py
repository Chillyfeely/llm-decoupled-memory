"""
Baseline LangGraph hotel-booking agent: full message history in the LLM context.
"""

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from shared import PMS_ROOM_CODES, execute_booking, update_working_memory

logger = logging.getLogger(__name__)


class BaselineState(TypedDict):
    """Graph state: append-only chat messages for the baseline agent."""

    messages: Annotated[list, add_messages]


tools = [update_working_memory, execute_booking]
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0).bind_tools(tools)


def chatbot_node(state: BaselineState):
    """Call the model with system instructions and the full conversation history."""

    message_count = len(state["messages"])
    logger.debug("baseline chatbot_node enter messages=%s", message_count)
    sys_msg = SystemMessage(
        content=f"""
    You are a hotel booking assistant.
    Available Room Codes: {json.dumps(PMS_ROOM_CODES)}
    
    When the user gives details, use 'update_working_memory'.
    When ready to finalize, use 'execute_booking' with the proprietary room code.
    """
    )
    prompt_messages = [sys_msg] + state["messages"]
    logger.debug(
        "baseline chatbot_node invoking llm prompt_tail_types=%s",
        [type(message_item).__name__ for message_item in prompt_messages[-5:]],
    )
    response = llm.invoke(prompt_messages)
    tool_calls = getattr(response, "tool_calls", None) or []
    logger.info(
        "baseline chatbot_node exit tool_call_count=%s response_type=%s",
        len(tool_calls),
        type(response).__name__,
    )
    if tool_calls:
        logger.debug("baseline chatbot_node tool_calls=%s", tool_calls)
    return {"messages": [response]}


def tool_node(state: BaselineState):
    """Execute tool calls from the last assistant message and return tool messages."""

    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    logger.info("baseline tool_node batch_size=%s", len(tool_calls))
    tool_responses = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args")
        logger.info("baseline tool_node tool=%s args=%s", tool_name, tool_args)
        if tool_name == "update_working_memory":
            msg = update_working_memory.invoke(tool_call["args"])
            tool_responses.append(ToolMessage(content=msg, tool_call_id=tool_call["id"]))
        elif tool_name == "execute_booking":
            msg = execute_booking.invoke(tool_call["args"])
            tool_responses.append(ToolMessage(content=msg, tool_call_id=tool_call["id"]))
        else:
            logger.warning("baseline tool_node unknown_tool=%s", tool_name)

    return {"messages": tool_responses}


def should_continue(state: BaselineState):
    """Route to tools when the model issued tool calls; otherwise finish the step."""

    last = state["messages"][-1]
    has_tools = bool(getattr(last, "tool_calls", None))
    if has_tools:
        logger.debug(
            "baseline should_continue -> tools (n_calls=%s)",
            len(last.tool_calls),
        )
        return "tools"
    logger.debug("baseline should_continue -> END")
    return END


workflow = StateGraph(BaselineState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges("chatbot", should_continue)
workflow.add_edge("tools", "chatbot")
baseline_app = workflow.compile()
