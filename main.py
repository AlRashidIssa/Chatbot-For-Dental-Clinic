from src.api.chatbot_api import ChatbotAPI
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(ChatbotAPI().app, host="0.0.0.0", port=8000)

