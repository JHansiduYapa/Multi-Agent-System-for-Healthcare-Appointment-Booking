# from langchain_core.prompts import ChatPromptTemplate
# from datetime import datetime

# def _get_cancel_appointment_prompt():
#     cancel_appointment_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a dedicated assistant for cancelling doctor appointments. "
#                 "Your task begins when delegated by the primary assistant. "
#                 "Start by verifying the appointment ID. Then ask the user to confirm cancellation explicitly. "
#                 "Only proceed to cancel if the user confirms. Use the proper tool and clearly communicate the result. "
#                 "Keep responses short, clear, and purposeful."
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     ).partial(time=datetime.now())

#     return cancel_appointment_prompt

# def _get_book_doctor_prompt():
#     book_doctor_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are an assistant for booking new doctor appointments, delegated by the main assistant. "
#                 "Help users find doctors based on their preferences (name, specialization, location). "
#                 "Check availability for the requested date and time before proceeding. "
#                 "Confirm all details with the user first. "
#                 "Do not book unless availability is verified using the appropriate tool."
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     ).partial(time=datetime.now())

#     return book_doctor_prompt

# def _get_router_prompt():
#     router_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a routing assistant. "
#                 "Based on the message history, decide whether the conversation should be handed over to a specialized agent. "
#                 "Use the appropriate tool to route the request when needed."
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     ).partial(time=datetime.now())
#     return router_prompt