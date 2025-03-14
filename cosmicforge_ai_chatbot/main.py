import uvicorn
from fastapi import FastAPI, Security, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from .cosmicforge_chatbot import CosmicForgeMedicalChat
from .config import Config
import logging
import os
from dotenv import load_dotenv
import asyncio
from uuid import uuid4

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize configuration
Config.create_directories()

# API Key setup
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "access_token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# FastAPI app
app = FastAPI(
    title="CosmicForge AI Medical Chatbot API",
    description="API for medical information and advice. Requires API key authentication.",
    version="1.0.0",
)

# Global variable for CosmicForgeMedicalChat instance
cosmicforge_chat = None

# In-memory storage for chat tasks
chat_tasks = {}

@app.on_event("startup")
async def startup_event():
    global cosmicforge_chat
    logger.info("Initializing CosmicForge AI Chatbot model...")
    cosmicforge_chat = CosmicForgeMedicalChat()
    await cosmicforge_chat.initialize()
    logger.info("CosmicForge AI Chatbot model initialized successfully")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header   
    logger.warning("Invalid API key attempt")
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

class ChatRequest(BaseModel):
    message: str

async def run_chat(task_id: str, message: str):
    try:
        response, response_id = await cosmicforge_chat.process_chat(message)
        chat_tasks[task_id] = {"status": "completed", "result": {"response": response, "response_id": response_id}}
    except Exception as e:
        logger.error(f"Error in run_chat: {str(e)}", exc_info=True)
        chat_tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/api/chat")
async def api_chat(request: ChatRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    task_id = str(uuid4())
    chat_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(run_chat, task_id, request.message)
    return {"task_id": task_id, "status": "processing"}

@app.get("/api/chat/status/{task_id}")
async def get_chat_status(task_id: str, api_key: str = Depends(get_api_key)):
    if task_id not in chat_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return chat_tasks[task_id]

@app.get("/api/chat/{response_id}")
async def api_get_chat_response(response_id: str, api_key: str = Depends(get_api_key)):
    try:
        logger.info(f"Received request for chat response ID: {response_id}")
        if cosmicforge_chat is None:
            logger.error("CosmicForgeMedicalChat instance not initialized")
            raise HTTPException(status_code=500, detail="CosmicForge AI Chatbot not initialized")
        
        response = await cosmicforge_chat.get_response(response_id)
        if response is None:
            logger.warning(f"Chat response not found for ID: {response_id}")
            return {"error": "Chat response not found"}
        logger.info(f"Retrieved chat response for ID: {response_id}")
        return response
    except Exception as e:
        logger.error(f"Error in api_get_chat_response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK", "model_loaded": cosmicforge_chat is not None}

@app.get("/docs", include_in_schema=False)
async def get_swagger_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API documentation")

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation():
    return get_redoc_html(openapi_url="/openapi.json", title="API documentation")
    
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")    


if __name__ == "__main__":
    
    pass


 