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
from agent.utils.llm import _get_llm,dummy_token_counter
from agent.utils.tools import book_appointment,cancel_appointment,search_for_appointment,check_doctor_availability,search_for_doctor,new_booking_assistant,cancel_booking_assistant
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

# state of the graph
class State(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]


# call the model in input
def new_booking_assistant_node(state: State, config: RunnableConfig) -> Dict[Any]:
    """Process input and returns output.can use runtime configuration to alter behavior.
    """
    llm_new_booking = _get_llm()
    llm_new_booking = llm_new_booking.bind_tools([])
    llm_new_booking_with_tools = llm_new_booking.bind_tools([book_appointment,search_for_doctor,check_doctor_availability])
    response = llm_new_booking_with_tools.invoke(state["messages"])
    return {"messages": response}

def router_model(state: State)  -> Command[Literal["cancel_booking_assistant", "new_booking_assistant",END]]:
    """Route to correct worker agent, You only routing the conversation.
    """
    llm_router = _get_llm()
    llm_router = llm_router.bind_tools([])
    llm_router_with_tools = llm_router.bind_tools([cancel_booking_assistant,new_booking_assistant])

    # trim the messages 
    # trim_messages(
    #     state['messages'],
    #     # Keep the last <= n_count tokens of the messages.
    #     strategy="last",
    #     # Remember to adjust based on your model
    #     # or else pass a custom token_counter
    #     token_counter=count_tokens_approximately,
    #     # Most chat models expect that chat history starts with either:
    #     # (1) a HumanMessage or
    #     # (2) a SystemMessage followed by a HumanMessage
    #     # Remember to adjust based on the desired conversation
    #     # length
    #     max_tokens=45,
    #     # Most chat models expect that chat history starts with either:
    #     # (1) a HumanMessage or
    #     # (2) a SystemMessage followed by a HumanMessage
    #     start_on="human",
    #     # Most chat models expect that chat history ends with either:
    #     # (1) a HumanMessage or
    #     # (2) a ToolMessage
    #     end_on=("human", "tool"),
    #     # Usually, we want to keep the SystemMessage
    #     # if it's present in the original history.
    #     # The SystemMessage has special instructions for the model.
    #     include_system=True,
    #     allow_partial=False,
    # )

    # make a custom prompt to instruct llm
    custom_prompt = {
    "role": "system",
    "content": "You are a routing assistant. "
                "Based on the message history, decide whether the conversation should be handed over to a specialized agent. "
                "Use the appropriate tool to route the request."
                "you can not use other tools on booking and cancelling."
    }
    messages_with_prompt = [custom_prompt] + state["messages"]

    # before pass to llm trim messsages to reduce token usage
    messages_with_prompt = trim_messages(
                                        messages_with_prompt,
                                        max_tokens=100,
                                        strategy="last",
                                        token_counter=dummy_token_counter,
                                        # Most chat models expect that chat history starts with either:
                                        # (1) a HumanMessage or
                                        # (2) a SystemMessage followed by a HumanMessage
                                        start_on="human",
                                        # Usually, we want to keep the SystemMessage
                                        # if it's present in the original history.
                                        # The SystemMessage has special instructions for the model.
                                        include_system=True,
                                        allow_partial=False,
                                    )

    # generate response
    response = llm_router_with_tools.invoke(messages_with_prompt)

    # if the router think to navigate to agent
    if hasattr(response, "tool_call") or hasattr(response, "tool_calls"):
        # get the tool name and route to relavant node
        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]

            # if not relavant tool called
            if tool_name not in ["cancel_booking_assistant", "new_booking_assistant"]:
                return Command(
                        # next node to be executed next
                        goto=END,
                        # 
                        update={"messages": [ToolMessage(content="not correct tool try to route correct specialized agent by cancel_booking_assistant,new_booking_assistant",tool_call_id=tool_id),HumanMessage(
                            content="The last tool call raised an exception. Try calling a correct tool again. Do not repeat mistakes."
                                ),]}
                )

            # route to relavant node
            return Command(
                # next node to be executed next
                goto=tool_name,
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
    llm_cancel_booking = llm_cancel_booking.bind_tools([])
    llm_cancel_booking_with_tools = llm_cancel_booking.bind_tools([cancel_appointment,search_for_appointment])
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
