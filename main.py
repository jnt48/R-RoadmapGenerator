from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- Import the middleware
from pydantic import BaseModel
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

# Create FastAPI instance
app = FastAPI(title="Roadmap Generator API", version="1.0.0")

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ['http://localhost:5173'] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the request payload
class RoadmapRequest(BaseModel):
    project_title: str
    project_description: str
    start_date: str  # You can also use datetime.date if preferred
    duration_months: int
    additional_notes: str = ""

# Pydantic model for the response
class RoadmapResponse(BaseModel):
    roadmap: str

def generate_roadmap(title: str, description: str, start: str, duration: int, notes: str) -> str:
    """
    Uses Gemini AI to generate a detailed roadmap for the provided project details.
    """
    prompt = f"""
    You are a seasoned project management expert and strategic planner.
    Generate a very detailed roadmap for a project with the following details:

    Project Title: {title}
    Project Description: {description}
    Start Date: {start}
    Estimated Duration: {duration} months
    Additional Notes: {notes}

    Your roadmap should include:
    - A clear timeline with phases or milestones
    - Specific tasks and objectives for each phase
    - Key performance indicators and checkpoints
    - Suggested resources and risk mitigation strategies
    - Final deliverables and review points

    Please ensure the roadmap is comprehensive, practical, and actionable for project managers and teams.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt])
    return response.text

@app.post("/roadmap", response_model=RoadmapResponse)
def roadmap_endpoint(request: RoadmapRequest):
    if not request.project_title or not request.project_description:
        raise HTTPException(status_code=400, detail="Project title and description are required.")
    
    try:
        roadmap_text = generate_roadmap(
            title=request.project_title,
            description=request.project_description,
            start=request.start_date,
            duration=request.duration_months,
            notes=request.additional_notes
        )
        return RoadmapResponse(roadmap=roadmap_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
