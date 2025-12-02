# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import router logic + global history + session id
from router_logic import process_message, conversation_history, SESSION_ID

# ======================================================
# FASTAPI APP INITIALIZATION
# ======================================================
app = FastAPI(title="Intelligent Chatbot API")

# ======================================================
# ENABLE CORS (Cross-Origin Resource Sharing)
# This allows frontend (JS/HTML) to access backend APIs even if on a different port.
# In development, allow all origins.
# In production, restrict to your frontend domain.
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (frontend)
    allow_methods=["*"],      # Allow GET, POST, PUT, DELETE
    allow_headers=["*"],      # Allow all headers
)

# ======================================================
# SERVE STATIC FRONTEND FILES
# This allows files in /frontend to be accessible via /static/...
# Example: index.html, style.css, app.js
# ======================================================
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ======================================================
# SERVE MAIN HTML PAGE
# Visiting http://localhost:8000/ will return index.html
# ======================================================
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

# ======================================================
# PYDANTIC MODEL: CHAT REQUEST
# This ensures POST /chat request must contain JSON:
# { "message": "user message here" }
# ======================================================
class Query(BaseModel):
    message: str

# ======================================================
# CHAT API ENDPOINT
# Handles the main chat logic:
# 1. Receives user message
# 2. Sends to intelligent router (process_message)
# 3. Returns JSON: session_id, tool_used, response
# ======================================================
@app.post("/chat")
def chat(q: Query):
    """
    POST /chat
    Request: { "message": "..." }
    Response: { "session_id": "...", "tool": "...", "response": "..." }
    """
    return process_message(q.message)

# ======================================================
# FULL HISTORY API
# Returns all past conversations for this session.
# Each entry contains:
# - session_id
# - user_message
# - bot_response
# - tool_used
# Useful for Postman testing or analytics.
# ======================================================
@app.get("/history")
def history():
    """
    GET /history
    Response: [
        {
            "session_id": "...",
            "user_message": "...",
            "bot_response": "...",
            "tool_used": "..."
        },
        ...
    ]
    """
    return conversation_history

# ======================================================
# LAST MESSAGE ONLY
# Returns only the most recent chat entry
# Useful if frontend only wants the latest response
# ======================================================
@app.get("/history/latest")
def last():
    """
    GET /history/latest
    Response:
        {
            "session_id": "...",
            "user_message": "...",
            "bot_response": "...",
            "tool_used": "..."
        }
    """
    if len(conversation_history) == 0:
        return {"message": "No history yet"}
    return conversation_history[-1]
