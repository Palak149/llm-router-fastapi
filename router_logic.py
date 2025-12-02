import torch, re, random, uuid, datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import Tool

# ======================================================
# FASTAPI APP INITIALIZATION
# ======================================================
app = FastAPI()

# Enable cross-origin requests (frontend → backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# GLOBAL SESSION ID (used to track each chat session)
# ======================================================
SESSION_ID = str(uuid.uuid4())

# ======================================================
# CONVERSATION HISTORY STORAGE
# This will store every chat step as a dictionary:
# {
#     "session_id": "...",
#     "timestamp": "...",
#     "user_message": "...",
#     "bot_response": "...",
#     "tool_used": "..."
# }
# ======================================================
conversation_history = []


# ======================================================
# LOAD MODELS 
# ======================================================

# 1️⃣ Small embedding model for semantic similarity routing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2️⃣ Qwen tiny instruct model (0.5B) for generating replies
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,     # saves memory
    device_map="cpu"               # runs on CPU
)

# 3️⃣ LangChain memory to store the AI conversation for internal use
memory = ConversationBufferMemory(return_messages=True)


# ======================================================
# CLEAN RESPONSE (removes special tokens)
# ======================================================
def clean(text):
    """Remove unwanted formatting & tokens from output."""
    if not text:
        return ""
    text = text.replace("|", " ")
    text = re.sub(r"(User|Assistant):", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ======================================================
# BASE LLM MODEL CALL
# Generates text using Qwen based on the query
# ======================================================
def llm(query):
    prompt = f"User: {query}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    return clean(tokenizer.decode(out[0], skip_special_tokens=True))


# ======================================================
# TOOL FUNCTIONS
# Each tool provides a specialized response
# ======================================================

# Tool 1: Suicide/Crisis support message
def suicide_tool(_):
    return ("I'm really sorry you're feeling this way. "
            "You deserve support. Please talk to someone you trust "
            "or contact emergency services or a local helpline immediately.")

# Tool 2: Random student marks generator
def marks_tool(_):
    subs = ["Math", "Physics", "Chemistry", "English", "Biology"]
    marks = {s: random.randint(40, 100) for s in subs}
    total = sum(marks.values())
    pct = round(total / len(subs), 2)

    reply = "\n".join([f"{s}: {m}/100" for s, m in marks.items()])
    return f"{reply}\nTotal: {total}/500\nPercentage: {pct}%"


# ======================================================
# TOOLS DICTIONARY
# Maps tool name → actual tool function
# ======================================================
tools = {
    "SuicideHelp": Tool(
        name="SuicideHelp",
        func=suicide_tool,
        description="Handles crisis or suicidal intent messages"
    ),
    "PositivePrompt": Tool(
        name="PositivePrompt",
        func=lambda x: llm(x),
        description="Comfort & motivational response"
    ),
    "NegativePrompt": Tool(
        name="NegativePrompt",
        func=lambda x: llm(x),
        description="Anxiety/worry-related response"
    ),
    "StudentMarks": Tool(
        name="StudentMarks",
        func=marks_tool,
        description="Random student marks generator"
    ),
}


# ======================================================
# ROUTER FUNCTION
# Detects user's intent → selects correct tool
# ======================================================
def route(user):
    text = user.lower()

    # Crisis detection
    if any(x in text for x in ["kill myself", "want to die", "suicide", "end my life"]):
        return "SuicideHelp"

    # Stress detection
    if any(x in text for x in ["stressed", "pressure", "tired", "overwhelmed"]):
        return "PositivePrompt"

    # Anxiety / worry detection
    if any(x in text for x in ["worried", "scared", "anxious", "fear"]):
        return "NegativePrompt"

    # Marks / scores
    if "mark" in text or "score" in text or "result" in text:
        return "StudentMarks"

    # Default tool if no match
    return "PositivePrompt"


# ======================================================
# MAIN CHAT HANDLER API
# ======================================================
@app.post("/chat")
def process_message(user: str):
    """
    Main chat endpoint: Receives a user message and returns
    the tool used + AI response.
    """

    # 1. Route to correct tool
    tool_name = route(user)

    # 2. Execute the tool function
    bot_response = tools[tool_name].func(user)

    # 3. Store the conversation in LangChain memory
    memory.chat_memory.add_user_message(user)
    memory.chat_memory.add_ai_message(bot_response)

    # 4. Save history for Postman
    conversation_history.append({
        "session_id": SESSION_ID,
        "timestamp": datetime.datetime.now().isoformat(),
        "user_message": user,
        "bot_response": bot_response,
        "tool_used": tool_name
    })

    # 5. Return clean JSON
    return {
        "session_id": SESSION_ID,
        "tool": tool_name,
        "response": bot_response
    }


# ======================================================
# API TO FETCH FULL CHAT HISTORY
# ======================================================
@app.get("/history")
def get_history():
    """
    Returns the entire conversation history:
    - session id
    - timestamp
    - user message
    - bot response
    - tool used
    """
    return conversation_history
