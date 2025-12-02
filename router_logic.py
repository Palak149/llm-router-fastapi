import torch, re, random, uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import Tool

# ======================================================
# GLOBALS
# ======================================================

# Unique session identifier for this chat session
SESSION_ID = str(uuid.uuid4())

# Stores every chat step in memory (user + AI + tool used)
conversation_history = []

# ======================================================
# LOAD MODELS
# ======================================================

# Embedding model to compute semantic similarity for intelligent routing
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Small Qwen LLM (0.5B) for generating responses
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,   # Use half-precision to save memory
    device_map="cpu"             # Run on CPU
)

# LangChain conversation memory to store AI/user messages internally
memory = ConversationBufferMemory(return_messages=True)

# ======================================================
# UTILITIES
# ======================================================

def clean(text):
    """
    Clean AI output:
    - Remove | symbols
    - Remove "User:" / "Assistant:" labels
    - Remove extra spaces
    """
    if not text:
        return ""
    text = text.replace("|", " ")
    text = re.sub(r"(User|Assistant):", "", text)
    return re.sub(r"\s+", " ", text).strip()


def llm(query):
    """
    Generate response using Qwen LLM
    """
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
# ======================================================

def suicide_tool(_):
    """
    Crisis / suicide support message
    """
    return (
        "I'm really sorry you're feeling this way. "
        "Please reach out to someone you trust or your local emergency services."
    )


def marks_tool(_):
    """
    Random student marks generator
    """
    subs = ["Math", "Physics", "Chemistry", "English", "Biology"]
    marks = {s: random.randint(40, 100) for s in subs}
    total = sum(marks.values())
    pct = round(total / len(subs), 2)

    reply = "\n".join([f"{s}: {m}/100" for s, m in marks.items()])
    return f"{reply}\nTotal: {total}/500\nPercentage: {pct}%"

# Tools dictionary: maps tool name → function + description
tools = {
    "SuicideHelp": Tool(
        name="SuicideHelp",
        func=suicide_tool,
        description="Crisis support for suicidal thoughts or emergencies."
    ),
    "PositivePrompt": Tool(
        name="PositivePrompt",
        func=lambda x: llm(x),
        description="Motivational or comforting responses for stressed users."
    ),
    "NegativePrompt": Tool(
        name="NegativePrompt",
        func=lambda x: llm(x),
        description="Responses for anxiety, worry, or fear."
    ),
    "StudentMarks": Tool(
        name="StudentMarks",
        func=marks_tool,
        description="Generates random student marks or scores."
    ),
}

# ======================================================
# INTELLIGENT ROUTER
# ======================================================

def route(user: str):
    """
    Intelligent routing function:
    - Uses semantic embeddings + conversation context
    - Chooses the most relevant tool automatically
    - Does NOT expose similarity scores
    """

    # 1. Get user input and clean
    user_message = user.strip()

    # 2. Fetch recent conversation context (last 3 user + 3 AI messages)
    context_messages = memory.chat_memory.messages[-6:]
    context_text = " ".join([msg.content for msg in context_messages])

    # 3. Combine context + current message
    combined_text = f"{context_text} {user_message}".strip()

    # 4. Encode user+context to embedding
    user_embedding = embedder.encode(combined_text, convert_to_tensor=True)

    # 5. Compare with each tool description (cosine similarity)
    best_tool = "PositivePrompt"  # default
    best_score = -1

    for tool_name, tool_obj in tools.items():
        tool_embedding = embedder.encode(tool_obj.description, convert_to_tensor=True)
        score = util.cos_sim(user_embedding, tool_embedding).item()
        if score > best_score:
            best_score = score
            best_tool = tool_name

    # 6. Fallback: low similarity → PositivePrompt
    if best_score < 0.4:
        best_tool = "PositivePrompt"

    return best_tool

# ======================================================
# MAIN CHAT HANDLER
# ======================================================

def process_message(user: str):
    """
    Main function to process user input:
    - Determines the best tool using intelligent routing
    - Executes the selected tool
    - Stores conversation in LangChain memory
    - Stores conversation in global history
    - Returns clean JSON: session_id, tool_used, response
    """

    # 1. Route to correct tool
    tool_name = route(user)

    # 2. Execute the tool function
    bot_response = tools[tool_name].func(user)

    # 3. Store conversation in LangChain memory
    memory.chat_memory.add_user_message(user)
    memory.chat_memory.add_ai_message(bot_response)

    # 4. Store conversation in global history
    conversation_history.append({
        "session_id": SESSION_ID,
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
