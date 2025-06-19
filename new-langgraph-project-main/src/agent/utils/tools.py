from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage,ToolMessage
from langchain_core.messages import ToolMessage 
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field
from typing import Literal

# Doctor appointment tools
def book_appointment(doctor_name: str, date: str, time: str):
    """Book a new doctor appointment"""
    return f"Appointment booked with Dr. {doctor_name} on {date} at {time}."

def reschedule_appointment(appointment_id: str, new_date: str, new_time: str):
    """Reschedule an existing doctor appointment"""
    return f"Appointment {appointment_id} rescheduled to {new_date} at {new_time}."

def cancel_appointment(appointment_id: str):
    """Cancel an existing doctor appointment"""
    return f"Appointment {appointment_id} has been cancelled."

def search_doctor(specialization: str = "", location: str = ""):
    """Search for doctors by specialization and/or location"""
    return f"Found doctors matching specialization='{specialization}' and location='{location}'."

def search_appointment(patient_id: str = "", doctor_name: str = "", date: str = ""):
    """Search for existing appointments by patient, doctor, or date"""
    return f"Found appointments for patient_id='{patient_id}', doctor='{doctor_name}', date='{date}'."

class new_booking_assistant(BaseModel):
    """If user needs to book a new appointment with a doctor"""
    reason: str = Field(description="Reason why the request should be routed")

class cancel_booking_assistant(BaseModel):
    """If user needs to cancel an appointment"""
    reason: str = Field(description="Reason why the request should be routed")