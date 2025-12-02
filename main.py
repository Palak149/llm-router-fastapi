from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from router_logic import process_message   # <-- Main routing + LLM logic
import uuid, datetime

# FASTAPI APPLICATION INITIALIZATION
# Creates a new FastAPI application instance.
# All API endpoints (/chat, /history, etc.) will be attached to this app.
app = FastAPI()

# GLOBAL SESSION ID
# A unique identifier for this session.
# Helps track which chat history belongs to which conversation.
# This ID remains the same until the server restarts.
SESSION_ID = str(uuid.uuid4())

# CONVERSATION HISTORY STORAGE
# This list will store every conversation turn in the following format:
#
# {
#   "session_id": "...unique id...",
#   "user_message": "What the user typed",
#   "bot_response": "What the bot returned",
#   "tool_used": "Name of tool selected by router_logic"
# }
#
# This is kept in-memory (RAM) only. It resets when server restarts.
conversation_history = []

# ENABLE CORS
# CORS allows the frontend (running on another port) to call this API.
# Without CORS, browsers block cross-domain API requests.
#
# allow_origins=["*"] means ANY frontend can access it.
# Good for development, but for production, restrict to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all domains (frontend can be anywhere)
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allow all custom headers
)

# STATIC FILES (Frontend Hosting)
# This exposes your frontend folder at /static path.
# So index.html, CSS, JS inside `frontend/` can be loaded by browser.
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

# SERVE MAIN INDEX.HTML FILE
# When a user opens http://localhost:8000/
# FastAPI will automatically return frontend/index.html.
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

# Pydantic Model: Defines POST /chat request format
# Ensures that incoming POST requests MUST contain:
# {
#    "message": "some text"
# }
class Query(BaseModel):
    message: str

# CHAT ENDPOINT → MAIN INTERACTION WITH AI
# This is the heart of your backend API.
# 1. Receives message from frontend.
# 2. Sends it to process_message() in router_logic.py.
#    - router_logic selects correct tool
#    - tool produces response
# 3. Saves the finalized result in conversation history.
# 4. Returns response back to frontend.

@app.post("/chat")
def chat(q: Query):

    # Extract user input
    user_msg = q.message

    # Send text to router logic → returns {tool, response}
    result = process_message(user_msg)

    # Extract details returned from router logic
    tool_used = result.get("tool")
    bot_response = result.get("response")

    # Build history entry for this chat turn
    history_entry = {
        "session_id": SESSION_ID,
        "user_message": user_msg,
        "bot_response": bot_response,
        "tool_used": tool_used
    }

    # Save the entry in local RAM
    conversation_history.append(history_entry)

    # Return actual chatbot reply to frontend
    return result

# API: GET FULL CHAT HISTORY
# Used by Postman or frontend to inspect entire conversation.
#
# Example response:
# [
#   {
#     "session_id": "...",
#     "timestamp": "...",
#     "user_message": "hi",
#     "bot_response": "Hello!",
#     "tool_used": "PositivePrompt"
#   },
#   ...
# ]
#
# This helps with debugging, audits, logs, analytics, etc.
@app.get("/history")
def get_history():
    return conversation_history

# API: GET MOST RECENT CHAT ENTRY
# Returns the last message pair
# Useful for cases where frontend only needs the latest reply
# instead of full history.
@app.get("/history/latest")
def get_last_history_entry():
    if len(conversation_history) == 0:
        return {"message": "No conversation history yet."}

    return conversation_history[-1]
