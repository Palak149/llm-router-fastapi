import torch, re, random, uuid, datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import Tool

# ======================================================
# GLOBAL SESSION ID (used to track each chat session)
# ======================================================
SESSION_ID = str(uuid.uuid4())

# ======================================================
# LOAD MODELS 
# ======================================================

# Small embedding model for similarity routing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qwen tiny instruct model (0.5B) for generating replies
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,   # saves memory
    device_map="cpu"             # runs on CPU
)

# LangChain memory to store previous messages
memory = ConversationBufferMemory(return_messages=True)

# ======================================================
# CLEAN RESPONSE (removes special tokens)
# ======================================================
def clean(text):
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

# Tool 1: Crisis support message
def suicide_tool(_):
    return ("I'm really sorry you're feeling this way. "
            "You deserve support. Please talk to someone you trust "
            "or call emergency services immediately.")

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
# This MUST be outside functions 
# Maps tool name → actual tool
# ======================================================
tools = {
    "SuicideHelp": Tool(
        name="SuicideHelp",
        func=suicide_tool,
        description="Crisis support"
    ),
    "PositivePrompt": Tool(
        name="PositivePrompt",
        func=lambda x: llm(x),
        description="Comfort response for stressed users"
    ),
    "NegativePrompt": Tool(
        name="NegativePrompt",
        func=lambda x: llm(x),
        description="Anxiety/worry related response"
    ),
    "StudentMarks": Tool(
        name="StudentMarks",
        func=marks_tool,
        description="Random marks generator"
    ),
}

# ======================================================
# ROUTER
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

    # Default response
    return "PositivePrompt"

# ======================================================
# MAIN CHAT HANDLER
# This runs when frontend calls /chat
# ======================================================
def process_message(user):

    # 1. Find the correct tool using the router
    tool_name = route(user)

    # 2. Execute the tool
    tool_output = tools[tool_name].func(user)

    # 3. Store conversation in memory
    memory.chat_memory.add_user_message(user)
    memory.chat_memory.add_ai_message(tool_output)

    # 4. Return clean JSON to frontend
    return {
        "tool": tool_name,
        "response": tool_output
    }
