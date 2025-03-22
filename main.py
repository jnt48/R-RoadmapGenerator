from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Pydantic Model
class VideoURL(BaseModel):
    url: str
    language: str = "en"  # Default language is English

# Function to extract YouTube Video ID
def extract_video_id(youtube_url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if match:
        return match.group(1)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Please provide a valid video link."
        )

# Function to fetch transcript text
def fetch_transcript(video_id, language):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    except Exception:
        return None

def extract_transcript_text(youtube_video_url, language="en"):
    video_id = extract_video_id(youtube_video_url)
    transcript = fetch_transcript(video_id, language)
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript extraction failed.")
    return " ".join([entry["text"] for entry in transcript])

# Function to generate summary using Gemini AI
def generate_summary(transcript_text, target_language):
    prompt = f"""
    You are a YouTube video summarizer. Summarize the given transcript into key points 
    with full explanation in more than 500 words using Markdown format. Include emojis for readability. 

    If the transcript is not in English, translate it to English first before summarizing.

    Transcript:
    {transcript_text}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(prompt)

        print("🔹 Raw Gemini Response:", response)
        if hasattr(response, 'text'):
            return response.text
        elif isinstance(response, dict):
            return response.get("candidates", [{}])[0].get("content", "Summary generation failed.")
        else:
            return "Unexpected response format from Gemini API."
    except Exception as e:
        print("❌ Gemini API Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@app.post("/api/summarize")
async def summarize_video(video: VideoURL):
    try:
        print(f"📌 Received URL: {video.url}, Language: {video.language}")

        transcript = extract_transcript_text(video.url, video.language)
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcript extraction failed.")

        summary = generate_summary(transcript, target_language=video.language)
        print("📌 Generated Summary:", summary[:500])

        return {
            "summary": summary,
            "questions": [
                "What are the main points discussed in the video?",
                "How does this content relate to the broader context?",
                "What evidence is presented to support the main arguments?"
            ]
        }
    except HTTPException as e:
        print("❌ API Error:", str(e))
        raise e
    except Exception as e:
        print("❌ Unexpected Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Generate 5 MCQs with detailed prompt
@app.post("/api/generate-mcqs")
async def generate_mcqs(video: VideoURL):
    try:
        transcript = extract_transcript_text(video.url, video.language)
        if not transcript:
            raise HTTPException(status_code=400, detail="Transcript extraction failed.")

        prompt = f"""
        You are an expert MCQ generator. Create **5 high-quality multiple-choice questions (MCQs)** 
        based on the given video transcript. 

        - Each question should be **conceptual, not factual**.
        - Provide **4 options** per question.
        - Clearly indicate the **correct answer**.
        - Explain why the correct answer is right.
        - Keep questions **challenging yet understandable**.

        Example format:
        Q1: [Question]
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        Correct Answer: [Correct Option]
        Explanation: [Why this is correct]

        Transcript:
        {transcript}
        """

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        # Extracting MCQs from response
        mcqs = []
        current_mcq = None
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("Q"):
                if current_mcq:
                    mcqs.append(current_mcq)
                current_mcq = {"question": line, "options": [], "correct_answer": "", "explanation": ""}
            elif line.startswith(("A)", "B)", "C)", "D)")):
                current_mcq["options"].append(line)
            elif line.startswith("Correct Answer:"):
                current_mcq["correct_answer"] = line.split(":")[1].strip()
            elif line.startswith("Explanation:"):
                current_mcq["explanation"] = line.split(":")[1].strip()

        if current_mcq:
            mcqs.append(current_mcq)

        if len(mcqs) < 5:
            raise HTTPException(status_code=500, detail="Not enough MCQs generated.")

        return {"mcqs": mcqs}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
