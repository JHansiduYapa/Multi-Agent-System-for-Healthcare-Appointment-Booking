import sqlite3

DB_PATH = "src/agent/database/appointments.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def execute_query(query, params=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or [])
        conn.commit()
        return cursor.fetchall()
    finally:
        conn.close()
