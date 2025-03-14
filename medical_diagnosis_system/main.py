import uvicorn
from fastapi import FastAPI, Security, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from medical_diagnosis_system.medical_diagnosis import MedicalDiagnosis
from cosmicforge_ai_chatbot.config import Config
from pydantic import BaseModel
from cosmicforge_ai_chatbot.config import Config
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
    title="Medical Diagnosis API",
    description="API for medical diagnosis. Requires API key authentication.",
    version="1.0.0",
)

# Global variable for MedicalDiagnosis instance
medical_diagnosis = None

# In-memory storage for diagnosis tasks
diagnosis_tasks = {}

@app.on_event("startup")
async def startup_event():
    global medical_diagnosis
    logger.info("Initializing MedicalDiagnosis model...")
    medical_diagnosis = MedicalDiagnosis()
    await medical_diagnosis.initialize()
    logger.info("MedicalDiagnosis model initialized successfully")

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

class DiagnosisRequest(BaseModel):
    symptoms: str

async def run_diagnosis(task_id: str, symptoms: str):
    try:
        diagnosis, diagnosis_id = await medical_diagnosis.diagnose(symptoms)
        diagnosis_tasks[task_id] = {"status": "completed", "result": {"diagnosis": diagnosis, "diagnosis_id": diagnosis_id}}
    except Exception as e:
        logger.error(f"Error in run_diagnosis: {str(e)}", exc_info=True)
        diagnosis_tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/api/diagnose")
async def api_diagnose(request: DiagnosisRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    task_id = str(uuid4())
    diagnosis_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(run_diagnosis, task_id, request.symptoms)
    return {"task_id": task_id, "status": "processing"}

@app.get("/api/diagnosis/status/{task_id}")
async def get_diagnosis_status(task_id: str, api_key: str = Depends(get_api_key)):
    if task_id not in diagnosis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return diagnosis_tasks[task_id]

@app.get("/api/diagnosis/{diagnosis_id}")
async def api_get_diagnosis(diagnosis_id: str, api_key: str = Depends(get_api_key)):
    try:
        logger.info(f"Received request for diagnosis ID: {diagnosis_id}")
        if medical_diagnosis is None:
            logger.error("MedicalDiagnosis instance not initialized")
            raise HTTPException(status_code=500, detail="Medical diagnosis system not initialized")
        
        diagnosis = await medical_diagnosis.get_diagnosis(diagnosis_id)
        if diagnosis is None:
            logger.warning(f"Diagnosis not found for ID: {diagnosis_id}")
            return {"error": "Diagnosis not found"}
        logger.info(f"Retrieved diagnosis for ID: {diagnosis_id}")
        return diagnosis
    except Exception as e:
        logger.error(f"Error in api_get_diagnosis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK", "model_loaded": medical_diagnosis is not None}

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

 