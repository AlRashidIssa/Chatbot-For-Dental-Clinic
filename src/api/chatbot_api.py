from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, sys

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Import chatbot pipeline
from pipeline_chatbot.chatbot_pipeline import FullPipelineChatbot
from configs import load_config_from_yaml
from utils.monitors import HighLevelErrors

# Load configuration
config = load_config_from_yaml()

# Initialize FastAPI app
app = FastAPI()

# Mount static files for CSS/JS
app.mount("/static", StaticFiles(directory=f"{MAIN_DIR}/api/static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Pydantic model for request body
class QueryModel(BaseModel):
    query: str

# Initialize the chatbot pipeline
chatbot = FullPipelineChatbot(config=config)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(query_model: QueryModel):
    try:
        query = query_model.query
        response = chatbot.run(query=query)
        return {"response": response}
    except Exception as e:
        HighLevelErrors.error(f"Error in API: {e}")
        raise HTTPException(status_code=500, detail=str(e))