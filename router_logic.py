import torch, re, random, uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import Tool

# GLOBAL SESSION ID (used to track each chat session)
SESSION_ID = str(uuid.uuid4())

# CONVERSATION HISTORY STORAGE
conversation_history = []


# LOAD MODELS 

#  Small embedding model for semantic similarity routing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#  Qwen tiny instruct model (0.5B) for generating replies
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,     # saves memory
    device_map="cpu"               # runs on CPU
)

#  LangChain memory to store the AI conversation for internal use
memory = ConversationBufferMemory(return_messages=True)

# CLEAN RESPONSE (removes special tokens)
def clean(text):
    """Remove unwanted formatting & tokens from output."""
    if not text:
        return ""
    text = text.replace("|", " ")
    text = re.sub(r"(User|Assistant):", "", text)
    return re.sub(r"\s+", " ", text).strip()

# BASE LLM MODEL CALL
# Generates text using Qwen based on the query
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

# TOOL FUNCTIONS
# Each tool provides a specialized response

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

# TOOLS DICTIONARY
# Maps tool name → actual tool function
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

# ROUTER FUNCTION
# Detects user's intent → selects correct tool

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

# MAIN CHAT HANDLER API

def process_message(user: str):
    tool_name = route(user)
    bot_response = tools[tool_name].func(user)

    # Save conversation in memory
    memory.chat_memory.add_user_message(user)
    memory.chat_memory.add_ai_message(bot_response)

    # Save to global history
    conversation_history.append({
        "session_id": SESSION_ID,
        "user_message": user,
        "bot_response": bot_response,
        "tool_used": tool_name
    })

    return {
        "session_id": SESSION_ID,
        "tool": tool_name,
        "response": bot_response
    }
