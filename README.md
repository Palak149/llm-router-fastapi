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


llm-router-chat/
│
├─ main.py                 # FastAPI backend, serves frontend + APIs
├─ router_logic.py         # Chatbot logic, tools, intelligent routing, memory
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation
│
├─ frontend/               # Frontend files
│   ├─ index.html          # Main chat UI
└─ venv/                   # Python virtual environment (optional, created locally)


## Architecture

Backend Architecture: Intelligent LLM Router Chat
=================================================

+------------------+
|   Frontend UI    |
|  (index.html)    |
+--------+---------+
         |
         | POST /chat  (user message)
         v
+----------------------------+
|        main.py             |
| - FastAPI app              |
| - Defines endpoints:       |
|    /chat                   |
|    /history                |
|    /history/latest         |
| - Serves frontend files    |
+------------+---------------+
             |
             | Calls
             v
+----------------------------+
|      router_logic.py       |
| - Intelligent router       |
| - Tool definitions         |
| - LLM call (Qwen 0.5B)    |
| - Conversation memory      |
| - Global session ID        |
| - History storage          |
+------------+---------------+
             |
             v
+----------------------------+
| Tools / Handlers           |
| - SuicideHelp              |
| - PositivePrompt           |
| - NegativePrompt           |
| - StudentMarks             |
+----------------------------+

Data Flow / Routing Logic
-------------------------
1. User sends a message via POST /chat.
2. main.py calls process_message(user_message) in router_logic.py.
3. router_logic:
   a. Fetches recent conversation context from memory.
   b. Combines user message + context.
   c. Uses embedding model (SentenceTransformer) to semantically compare
      the combined text with each tool's description.
   d. Selects the most relevant tool automatically (highest similarity).
4. Executes selected tool:
   - Some tools call LLM to generate response.
   - Others are predefined (e.g., marks, suicide help).
5. Stores user message + AI response in:
   - LangChain conversation memory
   - Global in-memory history (conversation_history)
6. Returns JSON to frontend:
   {
       "session_id": "...",
       "tool": "ToolName",
       "response": "AI or tool response"
   }

History & Analytics
-------------------
- /history → Returns all past messages with tool used.
- /history/latest → Returns only the last message.
- Useful for Postman testing, debugging, or frontend display.


## Installation
```bash
1. Clone the repo:
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


