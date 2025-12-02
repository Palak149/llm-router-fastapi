from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import router logic + global history
from router_logic import process_message, conversation_history, SESSION_ID

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")


# Pydantic for chat request
class Query(BaseModel):
    message: str


# ---------------- CHAT API ----------------
@app.post("/chat")
def chat(q: Query):
    return process_message(q.message)


# ---------------- FULL HISTORY API ----------------
@app.get("/history")
def history():
    return conversation_history


# ---------------- LAST MESSAGE ONLY ----------------
@app.get("/history/latest")
def last():
    if len(conversation_history) == 0:
        return {"message": "No history yet"}
    return conversation_history[-1]
