"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import GenerateContentConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from agent.utils.llm import _get_llm
from agent.utils.tools import book_appointment,cancel_appointment,search_for_appointment,check_doctor_availability,search_for_doctor,new_booking_assistant,cancel_booking_assistant
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
#from agent.utils.prompts import _get_cancel_appointment_prompt, _get_router_prompt, _get_book_doctor_prompt

# state of the graph
class State(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]


# call the model in input
def new_booking_assistant_node(state: State, config: RunnableConfig) -> Dict[Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    llm_new_booking = _get_llm()
    llm_new_booking_with_tools = llm_new_booking.bind_tools([book_appointment,search_for_doctor,check_doctor_availability])

    # make a runnable with prompt template
    # new_booking_runnable = book_doctor_prompt | llm_new_booking_with_tools
    response = llm_new_booking_with_tools.invoke(state["messages"])

    return {"messages": response}

def router_model(state: State)  -> Command[Literal["cancel_booking_assistant", "new_booking_assistant",END]]:
    """Route to correct worker agent, You only routing the conversation.
    """
    llm_router = _get_llm()
    llm_router_with_tools = llm_router.bind_tools([cancel_booking_assistant,new_booking_assistant])

    # make a runnable with prompt template
    # router_runnable = router_prompt | llm_router_with_tools
    custom_prompt = {
    "role": "system",
    "content": "You are a routing assistant. "
                "Based on the message history, decide whether the conversation should be handed over to a specialized agent. "
                "Use the appropriate tool to route the request when needed."
                "you can not use other tools on booking and cancelling."
    }
    messages_with_prompt = [custom_prompt] + state["messages"]

    response = llm_router_with_tools.invoke(messages_with_prompt)

    # if the router think to navigate to agent
    if hasattr(response, "tool_call") or hasattr(response, "tool_calls"):
        # get the tool name and route to relavant node
        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]

            # # if not relavant tool called
            # if tool_name not in ["cancel_booking_assistant", "new_booking_assistant"]:
            #     return Command(
            #             # next node to be executed next
            #             goto="router_assistant",
            #             # 
            #             update={"messages": [ToolMessage(content="not correct tool try to route correct specialized agent by cancel_booking_assistant,new_booking_assistant",tool_call_id=tool_id)]}
            #     )

            # route to relavant node
            return Command(
                # next node to be executed next
                goto=tool_name,
                # state update 
                update={"messages": [],}
            )
    
    # if the tool is not called and the llm provide the output
    return Command(
                # next node to be executed next
                goto=END,
                # state update 
                update={"messages": response,}
            )

# make cancel appointment node
def cancel_booking_assistant_node(state: State) -> Dict[Any]:
    llm_cancel_booking = _get_llm()
    llm_cancel_booking_with_tools = llm_cancel_booking.bind_tools([cancel_appointment,search_for_appointment])

    # make a runnable with prompt template
    # cancel_booking_runnable = cancel_appointment_prompt | llm_cancel_booking_with_tools

    response = llm_cancel_booking_with_tools.invoke(state["messages"])
    return {"messages": response}


# Define the graph
graph_builder = StateGraph(State)
graph_builder.add_node("router_assistant", router_model)
graph_builder.add_node("new_booking_assistant", new_booking_assistant_node)
graph_builder.add_node("cancel_booking_assistant", cancel_booking_assistant_node)
graph_builder.add_node("new_booking_tools", ToolNode([book_appointment, search_for_doctor,check_doctor_availability]))
graph_builder.add_node("cancel_booking_tools", ToolNode([cancel_appointment, search_for_appointment]))

graph_builder.set_entry_point("router_assistant")

# Conditional edge: if tool call, go to "tools", else END
graph_builder.add_conditional_edges(
    "new_booking_assistant",
    tools_condition,
    {"tools": "new_booking_tools", "__end__": END}
)

# Conditional edge: if tool call, go to "tools", else END
graph_builder.add_conditional_edges(
    "cancel_booking_assistant",
    tools_condition,
    {"tools": "cancel_booking_tools", "__end__": END}
)

# After tools, return to LLM node
graph_builder.add_edge("new_booking_tools", "new_booking_assistant")
graph_builder.add_edge("cancel_booking_tools", "cancel_booking_assistant")
graph_builder.add_edge("new_booking_assistant", END)
graph_builder.add_edge("cancel_booking_assistant", END)

graph = graph_builder.compile()
