import os
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime

# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

# Import chatbot pipeline and other utilities
from pipeline_chatbot.chatbot_pipeline import FullPipelineChatbot
from save_pydantic.history_chat import ChatHistoryEdge
from configs import load_config_from_yaml
from utils.monitors import HighLevelErrors, APIOperation

# Load configuration
config = load_config_from_yaml()

# Pydantic model for request body
class QueryModel(BaseModel):
    query: str

class ChatbotAPI:
    """
    A FastAPI-based chatbot application with monitoring and logging.

    Attributes:
        app (FastAPI): The FastAPI application instance.
        chatbot (FullPipelineChatbot): The chatbot pipeline.
        chat_edge (ChatHistoryEdge): The database handler for chat history.
        templates (Jinja2Templates): Template engine for rendering HTML.
    """

    def __init__(self):
        """
        Initializes the FastAPI app, chatbot pipeline, and database handler.
        """
        self.app = FastAPI()
        self.chatbot = FullPipelineChatbot(config=config)
        self.chat_edge = ChatHistoryEdge()
        self.templates = Jinja2Templates(directory="templates")

        # Mount static files for CSS/JS
        self.app.mount("/static", StaticFiles(directory=f"{MAIN_DIR}/api/static"), name="static")

        # Define routes
        self.app.get("/", response_class=HTMLResponse)(self.read_root)
        self.app.post("/chat")(self.chat)

    async def read_root(self, request: Request) -> HTMLResponse:
        """
        Renders the chat interface.

        Args:
            request (Request): The incoming HTTP request.

        Returns:
            HTMLResponse: The rendered chat.html template.
        """
        return self.templates.TemplateResponse("chat.html", {"request": request})

    async def chat(self, query_model: QueryModel) -> dict:
        """
        Handles the chatbot query and saves the interaction to the database.

        Args:
            query_model (QueryModel): The user's query in the request body.

        Returns:
            dict: The chatbot's response.

        Raises:
            HTTPException: If an error occurs during processing.
        """
        try:
            query = query_model.query
            APIOperation.info(f"Received query: {query}")

            # Run the query through the chatbot pipeline
            response = self.chatbot.run(query=query)
            APIOperation.info(f"Generated response: {response}")

            # Save the chat interaction to the database
            saved_chat = self.chat_edge.save_chat_response(query=query, response=response)
            APIOperation.info(f"Saved chat record: ID={saved_chat.id}, Query={saved_chat.query}, Response={saved_chat.response}, Timestamp={saved_chat.timestamp}")

            # Return the response to the front-end
            return {"response": response}

        except Exception as e:
            HighLevelErrors.error(f"Error in API: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")


# Initialize the ChatbotAPI class
chatbot_api = ChatbotAPI()
# Expose the FastAPI app for integration with main.py
app = chatbot_api.app