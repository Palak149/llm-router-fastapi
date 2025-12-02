# Intelligent LLM Router Chat System

## Overview

This project is a **context-aware chatbot** built using a combination of:

- **Qwen 0.5B LLM** for text generation
- **Sentence Transformers** for semantic embeddings
- **LangChain** for conversation memory
- **FastAPI** for backend APIs
- **HTML/CSS/JS frontend** for chat interface

The chatbot intelligently selects the most relevant tool to respond to user input based on **context and intent**, without relying solely on keyword matching.

---

## Features

1. **Intelligent Routing**
   - Routes user messages to different tools based on context and semantic similarity.
   - Tools include:
     - `SuicideHelp` – crisis support
     - `PositivePrompt` – motivational/comfort responses
     - `NegativePrompt` – anxiety/worry responses
     - `StudentMarks` – random marks generator

2. **Conversation Memory**
   - Uses **LangChain** to store the last few messages to understand context.
   - Keeps a **global in-memory history** for session tracking.

3. **APIs**
   - `POST /chat` – send user messages, receive bot responses
   - `GET /history` – fetch full chat history
   - `GET /history/latest` – fetch last message only

4. **Frontend**
   - Simple dark-themed chat UI
   - Displays tool used and bot response
   - Supports ENTER key and send button

---

## Folder Structure
project_root/
│
├─ main.py # FastAPI app, serves frontend and APIs
├─ router_logic.py # Chatbot logic, tools, intelligent routing
├─ frontend/
│ ├─ index.html # Chat UI
├─ requirements.txt # Python dependencies
└─ README.md


---

## Architecture

┌─────────────┐
│ User Input │
└──────┬──────┘
│
▼
┌─────────────┐
│ Frontend │
│ (HTML/JS) │
└──────┬──────┘
│ POST /chat
▼
┌─────────────┐
│ FastAPI │
│ Backend │
└──────┬──────┘
│
▼
┌────────────────────────────┐
│ Router Logic (router_logic.py) │
│ ┌──────────────────────────┐ │
│ │ Intelligent Router │ │
│ │ (Semantic + Context) │ │
│ └───────┬──────────────────┘ │
│ ▼ │
│ ┌──────────────────────────┐ │
│ │ Tools │ │
│ │ ┌──────────────────────┐ │ │
│ │ │ SuicideHelp │ │ │
│ │ │ PositivePrompt │ │ │
│ │ │ NegativePrompt │ │ │
│ │ │ StudentMarks │ │ │
│ │ └──────────────────────┘ │ │
│ └──────────────────────────┘ │
│ ▲ │
│ │ │
│ ┌──────────────────────────┐ │
│ │ LangChain Memory │ │
│ │ Stores last 3-6 messages │ │
│ └──────────────────────────┘ │
└──────────────────────────────┘
│
▼
┌─────────────┐
│ Response │
└──────┬──────┘
│
▼
┌─────────────┐
│ Frontend │
│ ChatBox │
└─────────────┘




**Flow Description:**

1. User enters a message in the frontend.
2. Frontend sends message via `POST /chat` to FastAPI backend.
3. Backend calls `process_message()`:
   - Retrieves recent conversation context from LangChain memory
   - Computes semantic embeddings
   - Routes message to the most relevant tool
   - Executes tool function or LLM
   - Stores user + AI response in memory
4. Backend returns JSON with:
   - `session_id`
   - `tool` used
   - `response` text
5. Frontend displays the tool used and the bot response.

---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/llm-router-chat.git
cd llm-router-chat

2.Create virtual environment:
python -m venv venv
venv\Scripts\activate          # Windows

3.Install dependencies:
pip install -r requirements.txt

4.Run the app:
uvicorn main:app --reload

5.Open browser:
http://localhost:8000

Usage

Type a message in the input box and press ENTER or click Send.

The bot will respond, showing the tool used and response.

Access full chat history via:
GET http://localhost:8000/history
GET http://localhost:8000/history/latest

Dependencies

fastapi
uvicorn
transformers
torch
sentence-transformers
langchain
langchain-community-tools
pydantic
fastapi-staticfiles
See requirements.txt for full versions.


