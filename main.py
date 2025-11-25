from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from router_logic import process_message  # Import your main LLM router logic

# ================================================
# CREATE FASTAPI APP
# ================================================
app = FastAPI()

# ================================================
# ENABLE CORS (Cross-Origin Resource Sharing)
# This allows your frontend (HTML/JS) to call the API
# even when both run on different ports (e.g., 5500 & 8000)
# ================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow ALL domains (good for development)
    allow_methods=["*"],      # Allow GET, POST, PUT, DELETE etc.
    allow_headers=["*"],      # Allow all request headers
)

# ================================================
# SERVE STATIC FRONTEND FILES
# 'frontend' folder contains index.html, CSS, JS
# /static/... will load files from /frontend directory
# ================================================
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

# ================================================
# SERVE MAIN HTML PAGE
# When user visits http://localhost:8000/
# FastAPI will return the index.html file
# ================================================
@app.get("/")
def root():
    return FileResponse("frontend/index.html")

# ================================================
# Pydantic model for POST /chat
# Ensures JSON request contains: { "message": "..." }
# ================================================
class Query(BaseModel):
    message: str  # The user's text input

# ================================================
# CHAT ENDPOINT
# The frontend sends the user's message here
# process_message() routes the query to the right tool
# Returns JSON: { tool:..., response:... }
# ================================================
@app.post("/chat")
def chat(q: Query):
    response = process_message(q.message)  # Run router logic
    return response  # FastAPI automatically converts to JSON
