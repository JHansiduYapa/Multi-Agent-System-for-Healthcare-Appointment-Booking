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
from agent.utils.tools import book_appointment,reschedule_appointment,cancel_appointment,search_doctor,search_appointment,new_booking_assistant,cancel_booking_assistant
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


# state of the graph
class State(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]


# call the model in input
def new_booking_assistant_node(state: State, config: RunnableConfig) -> Dict[Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config["configurable"]
    llm_new_booking = _get_llm()
    llm_new_booking_with_tools = llm_new_booking.bind_tools([book_appointment,search_doctor])
    response = llm_new_booking_with_tools.invoke(state["messages"])
    return {"messages": response}

def router_model(state: State, config: RunnableConfig)  -> Command[Literal["cancel_booking_assistant", "new_booking_assistant"]]:
    """Route to correct worker agent by calling relavant tool.

    """
    llm_router = _get_llm()
    llm_router_with_tools = llm_router.bind_tools([cancel_booking_assistant,new_booking_assistant])
    response = llm_router_with_tools.invoke(state["messages"])
    # if the router think to navigate to agent
    if hasattr(response, "tool_call") or hasattr(response, "tool_calls"):
        # get the tool name and route to relavant node
        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            
            # route to relavant node
            return Command(
                # next node to be executed next
                goto=tool_name,
                # state update 
                update={"messages": [], "dialog_state":tool_name}
            )
    
    return {"messages": response}

# make cancel appointment node
def cancel_booking_assistant_node(state: State, config: RunnableConfig) -> Dict[Any]:
    configuration = config["configurable"]
    llm_new_booking = _get_llm()
    llm_new_booking_with_tools = llm_new_booking.bind_tools([cancel_appointment,search_appointment])
    response = llm_new_booking_with_tools.invoke(state["messages"])
    return {"messages": response}


# Define the graph
graph_builder = StateGraph(State)
graph_builder.add_node("router_assistant", router_model)
graph_builder.add_node("new_booking_assistant", new_booking_assistant_node)
graph_builder.add_node("cancel_booking_assistant", cancel_booking_assistant_node)
graph_builder.add_node("tools", ToolNode([book_appointment,search_doctor]))

graph_builder.set_entry_point("router_assistant")

# Conditional edge: if tool call, go to "tools", else END
graph_builder.add_conditional_edges(
    "new_booking_assistant",
    tools_condition,
    {"tools": "tools", "__end__": END}
)

# After tools, return to LLM node
graph_builder.add_edge("tools", "new_booking_assistant")
graph_builder.add_edge("new_booking_assistant", END)
graph_builder.add_edge("cancel_booking_assistant", END)

graph = graph_builder.compile()
