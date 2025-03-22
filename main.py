from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("GOOGLE_API_KEY is not set in your environment variables.")

# Configure Gemini AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="EDITH Chatbot API", version="1.0.0")

# Enable CORS to allow frontend to interact with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models for the conversation
class ChatMessage(BaseModel):
    role: str  # "user" or "bot"
    message: str

class ChatRequest(BaseModel):
    user_message: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    bot_message: str
    history: List[ChatMessage]

def generate_humorous_response(prompt: str) -> str:
    """
    Uses Gemini AI to generate a humorous response in the persona of EDITH (in a Tony Stark style).
    """
    # Construct a prompt with instructions for humor and wit
    full_prompt = (
        "You are EDITH, an AI chatbot with the wit and sarcasm of Tony Stark. "
        "Your tone is humorous, lighthearted, and occasionally cheeky, but always insightful. "
        "Answer the user's query with clever remarks and a touch of irony. \n" +
        prompt
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([full_prompt])
    return response.text.strip()

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_request: ChatRequest):
    # Get the conversation history or initialize an empty list
    history = chat_request.history or []

    # Append the new user message to the history
    history.append(ChatMessage(role="user", message=chat_request.user_message))

    # Limit history to the last 100 messages
    if len(history) > 100:
        history = history[-100:]

    # Build a conversation prompt with a system instruction
    conversation_prompt = (
        "System: You are EDITH, an educational, professional, interactive, and witty chatbot with a Tony Stark flair. "
        "Respond humorously, provide insightful answers, and always add a clever remark. \n"
    )
    for chat in history:
        if chat.role == "user":
            conversation_prompt += "User: " + chat.message + "\n"
        else:
            conversation_prompt += "EDITH: " + chat.message + "\n"
    conversation_prompt += "EDITH: "  # Prompt for the next answer

    # Get the bot's response from Gemini AI
    bot_message = generate_humorous_response(conversation_prompt)

    # Append the bot response to the conversation history
    history.append(ChatMessage(role="bot", message=bot_message))

    # Return the bot's response along with the updated history
    return ChatResponse(bot_message=bot_message, history=history)

