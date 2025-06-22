from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage,ToolMessage
from langchain_core.messages import ToolMessage 
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel
from langchain_core.tools import tool
from typing import Optional, Dict, List
import sqlite3
import json
import os
from datetime import datetime

# Define your database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'appointments.db')


# Tool 1: Search for doctor
@tool
def search_for_doctor(name: Optional[str] = None) -> str:
    """
    Search for doctors by name (or return all if name is None).

    Args:
        name (Optional[str]): The partial or full name of the doctor.

    Returns:
        str: JSON-encoded list of matching doctors.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if name:
        cursor.execute("SELECT * FROM Doctor WHERE Doctor_Name LIKE ?", (f"%{name}%",))
    else:
        cursor.execute("SELECT * FROM Doctor")

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return json.dumps(results)


# Tool 2: Check doctor's availability
@tool
def check_doctor_availability(doctor_id: int, date: str, time: str) -> str:
    """
    Check if a doctor is available at a given date and time.

    Args:
        doctor_id (int): The ID of the doctor.
        date (str): Date of the appointment in 'YYYY-MM-DD'.
        time (str): Time of the appointment in 'HH:MM'.

    Returns:
        str: JSON with availability message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1 FROM Appointment
        WHERE Doctor_ID = ? AND Appointment_Date = ? AND Appointment_Time = ?
    """, (doctor_id, date, time))

    result = cursor.fetchone()
    conn.close()

    available = result is None
    return json.dumps({"available": available, "message": "Doctor is available" if available else "Doctor is not available"})


# Tool 3: Book appointment
@tool
def book_appointment(user_id: int, doctor_id: int, date: str, time: str) -> str:
    """
    Book an appointment with a doctor if available.

    Args:
        user_id (int): The ID of the patient.
        doctor_id (int): The ID of the doctor.
        date (str): Appointment date in 'YYYY-MM-DD'.
        time (str): Appointment time in 'HH:MM'.

    Returns:
        str: JSON with booking status.
    """
    try:
        # Parse and validate appointment datetime
        appointment_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        now = datetime.now()

        if appointment_datetime <= now:
            return json.dumps({"success": False, "message": "Appointment must be booked for a future time."})

        if not (8 <= appointment_datetime.hour < 22):
            return json.dumps({"success": False, "message": "Appointment time must be between 08:00 and 22:00."})

        # Connect to DB and check availability
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 1 FROM Appointment
            WHERE Doctor_ID = ? AND Appointment_Date = ? AND Appointment_Time = ?
        """, (doctor_id, date, time))
        
        if cursor.fetchone():
            conn.close()
            return json.dumps({"success": False, "message": "Doctor is not available at this date and time."})

        # Insert the new appointment
        cursor.execute("""
            INSERT INTO Appointment (Appointment_Time, Appointment_Date, Doctor_ID, Patient_ID)
            VALUES (?, ?, ?, ?)
        """, (time, date, doctor_id, user_id))

        conn.commit()
        conn.close()

        return json.dumps({"success": True, "message": "Appointment booked successfully."})

    except ValueError:
        return json.dumps({"success": False, "message": "Invalid date or time format. Use YYYY-MM-DD and HH:MM."})
    except Exception as e:
        return json.dumps({"success": False, "message": f"An error occurred: {str(e)}"})

# Tool 4: Search for an appointment by ID
@tool
def search_for_appointment(appointment_id: int) -> str:
    """
    Search for an appointment by its ID.

    Args:
        appointment_id (int): The ID of the appointment.

    Returns:
        str: JSON string with appointment details or failure message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT Appointment_ID, Appointment_Date, Appointment_Time, Doctor_ID, Patient_ID
        FROM Appointment
        WHERE Appointment_ID = ?
    """, (appointment_id,))
    
    result = cursor.fetchone()
    conn.close()

    if result:
        return json.dumps({
            "success": True,
            "appointment": {
                "appointment_id": result[0],
                "date": result[1],
                "time": result[2],
                "doctor_id": result[3],
                "patient_id": result[4]
            }
        })
    else:
        return json.dumps({"success": False, "message": "Appointment not found."})


# Tool 5: Cancel an appointment by ID
@tool
def cancel_appointment(appointment_id: int) -> str:
    """
    Cancel an existing appointment by its ID.

    Args:
        appointment_id (int): The ID of the appointment to cancel.

    Returns:
        str: JSON string with success/failure status and message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if appointment exists
    cursor.execute("SELECT 1 FROM Appointment WHERE Appointment_ID = ?", (appointment_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return json.dumps({"success": False, "message": "Appointment not found."})

    # Cancel appointment
    cursor.execute("DELETE FROM Appointment WHERE Appointment_ID = ?", (appointment_id,))
    conn.commit()
    conn.close()

    return json.dumps({"success": True, "message": "Appointment cancelled successfully."})

# Tool 5: Cancel an appointment by ID
@tool
def reschedule_appointment(appointment_id: int, date: str, time: str) -> str:
    """
    Reschedule an existing appointment to a new date and time if the doctor is available.

    Args:
        appointment_id (int): ID of the appointment to update.
        date (str): New date in 'YYYY-MM-DD' format.
        time (str): New time in 'HH:MM' format.

    Returns:
        str: JSON string with success/failure status and message.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get doctor ID for the existing appointment
    cursor.execute("SELECT Doctor_ID FROM Appointment WHERE Appointment_ID = ?", (appointment_id,))
    result = cursor.fetchone()
    if not result:
        conn.close()
        return json.dumps({"success": False, "message": "Appointment not found."})

    doctor_id = result[0]

    # Check if doctor is already booked at the new time (excluding current appointment)
    cursor.execute("""
        SELECT 1 FROM Appointment
        WHERE Doctor_ID = ? AND Appointment_Date = ? AND Appointment_Time = ? AND Appointment_ID != ?
    """, (doctor_id, date, time, appointment_id))

    if cursor.fetchone():
        conn.close()
        return json.dumps({"success": False, "message": "Doctor is not available at the requested time."})

    # Update appointment
    cursor.execute("""
        UPDATE Appointment
        SET Appointment_Date = ?, Appointment_Time = ?
        WHERE Appointment_ID = ?
    """, (date, time, appointment_id))

    conn.commit()
    conn.close()

    return json.dumps({"success": True, "message": "Appointment Rescheduled successfully."})



class new_booking_assistant(BaseModel):
    """based on conversation history,If user needs to book a new appointment with a doctor"""

class cancel_booking_assistant(BaseModel):
    """based on conversation history,If user needs to cancel an appointment"""

class general_hospital_assistant(BaseModel):
    """based on conversation history,If user needs information about hospital"""