**Multi-Agent System for Healthcare Appointment Booking**
---

A modular, extensible framework for managing healthcare appointment workflows using multiple AI agents, including:

- **Router Agent**: Directs user requests to specialized services (new bookings, cancellations, general inquiries).
- **RAG Agent**: Provides question-answering capabilities by retrieving context from PDF documents.
- **Tool Nodes**: Interfaces for booking, cancellation, doctor search, and availability checks.
- **Audio Output**: Converts AI responses to speech via ElevenLabs.

---

## State Transitions

This GIF illustrates how the router inspects message history and transitions between nodes:

![State Transitions](https://github.com/JHansiduYapa/Multi-Agent-System-for-Healthcare-Appointment-Booking/blob/main/src/short_video.gif)

---

## Features

1. **Custom Router**
   - Implements routing logic from scratch.
   - Routes user queries to appropriate agent nodes based on intent.
   - **Router Implementation**: Built with LangGraph’s command functions, it inspects the full message history at each turn to decide the next node (e.g., booking, cancellation, RAG or audio) and dynamically dispatches the conversation to the correct assistant.
   - **Router Code**:

     ```python
     def router_model(
         state: State
     ) -> Command[Literal[
         "cancel_booking_assistant",
         "new_booking_assistant",
         "general_hospital_assistant",
         "audio_output",
         END
     ]]:
         """Route to correct worker agent. You only route the conversation."""
         llm_router = _get_llm().bind_tools([])
         llm_router_with_tools = llm_router.bind_tools([
             cancel_booking_assistant,
             new_booking_assistant,
             general_hospital_assistant
         ])

         custom_prompt = {
             "role": "system",
             "content": (
                 "You are a routing assistant. Based on the message history, "
                 "decide whether the conversation should be handed over to a specialized agent. "
                 "Use the appropriate tool to route the request. If no tool is needed, reply without calling a tool. "
                 "You cannot use other tools for booking or cancelling."
             )
         }
         messages_with_prompt = trim_messages(
             [custom_prompt] + state["messages"],
             max_tokens=100,
             strategy="last",
             token_counter=dummy_token_counter,
             start_on="human",
             include_system=True,
             allow_partial=False,
         )

         response = llm_router_with_tools.invoke(messages_with_prompt)

         if hasattr(response, "tool_calls") and response.tool_calls:
             tool_call = response.tool_calls[0]
             tool_name = tool_call["name"]
             if tool_name not in [
                 "cancel_booking_assistant",
                 "new_booking_assistant",
                 "general_hospital_assistant"
             ]:
                 return Command(
                     goto=END,
                     update={
                         "messages": [
                             ToolMessage(
                                 content=(
                                     "Not a correct tool. "
                                     "Try routing to cancel_booking_assistant or new_booking_assistant."
                                 ),
                                 tool_call_id=tool_call["id"]
                             ),
                             HumanMessage(
                                 content=(
                                     "The last tool call raised an exception. "
                                     "Try calling a correct tool again. Do not repeat mistakes."
                                 )
                             ),
                         ]
                     }
                 )
             return Command(goto=tool_name)

         return Command(
             goto="audio_output",
             update={"messages": response}
         )
     ```

2. **RAG Agent**
   - Loads and indexes PDF documents (Scope of Services, Statement of Purpose).
   - Retrieves relevant context for concise, three-sentence answers.

3. **Booking & Cancel Agents**
   - Includes tools such as `book_appointment`, `cancel_appointment`, `search_for_doctor`, `check_doctor_availability`, and `search_for_appointment`.
   - Ensures safe, efficient, and accurate appointment handling for both booking and cancellation workflows.

4. **Audio Response**
   - Uses ElevenLabs API to convert text responses to high-quality MP3 audio.

5. **State Graph Architecture**
   - Defines nodes and edges for conversational flow using `langgraph`.
   - Easily extendable for new services and integrations.

---

## Usage

1. **Start the chatbot** and enter your query.  
2. **Router Agent** directs the request:  
   - New bookings → `new_booking_assistant`  
   - Cancellations → `cancel_booking_assistant`  
   - General inquiries → `general_hospital_assistant` (RAG)  
3. **RAG Agent** answers information requests with context from PDFs.  
4. **Audio Output** plays the response via ElevenLabs.

### Example Session Flow

```python
def rag_node(state: State) -> Dict[Any]:
    file_path = ".../Scope-of-Services-Statement-of-Purpose.pdf"
    pages = load_document(file_path)
    documents = split_text(pages)
    vectorstore = create_vectorstore(documents)
    tool = get_retriever_tool(vectorstore)

    question = state["messages"][-1].content
    context = tool.invoke({"query": question})

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\n"
        "Context: {context}"
    )

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    llm = _get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {'messages': response}
````

---

## Architecture Diagram

![Architecture Diagram](https://github.com/JHansiduYapa/Multi-Agent-System-for-Healthcare-Appointment-Booking/blob/main/src/graph.png)

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request:

1. Fork
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

