# llm-router-fastapi
LLM Router FastAPI

A lightweight FastAPI backend powered by a custom LLM Router, Qwen-0.5B, and multiple AI tools such as Suicide Prevention, Positive Dialogue, Anxiety Response, and Random Marks Generator.

This project includes:

 LLM Routing System
 FastAPI Backend
 Frontend Chat UI
 Tool-based Decision Router
 Conversation Memory
 Safe Handling of Sensitive Inputs
 Clean UI with dark theme


 Project Structure

 llm_router_fastapi/
│
├── main.py               # FastAPI backend server
├── router_logic.py       # LLM routing and tool logic
├── frontend/
│   ├── index.html        # Frontend chat interface
│
├── README.md
└── requirements.txt

 Features
 1. Smart Router

Routes user input to the correct tool:

Trigger keywords	Tool	Purpose
"suicide", “end my life”	- SuicideHelp,	Safe crisis support
"stressed", “pressure”-	PositivePrompt,	Comfort-focused AI
"worried", “fear”, “anxious”-	NegativePrompt,	Calm/anxiety support
"marks", “result”, “score”-	StudentMarks,	Random student marks generator

 2. Tools Used
 Suicide Help Tool
Provides safe, responsible support messages.

 Positive Tool
Uses Qwen model to generate comfort responses.

 Negative Tool
LLM generates anxiety-relief responses.

 Student Marks Tool
Generates random marks for 5 subjects.

3. Frontend UI

Chat box

Dark theme

Displays Tool Used + LLM Response

Enter key support

Error handling
