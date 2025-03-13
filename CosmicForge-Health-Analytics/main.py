import json
import os
import sys
import time
from typing import Dict, Any, List, Union
from datetime import datetime
import threading
import queue
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.responses import RedirectResponse

from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import uvicorn

from logger import get_logger, log_exception, setup_uncaught_exception_handler
from config import API_HOST, API_PORT, API_WORKERS, MONITORING_INTERVAL, API_KEY
from genetics import get_personalized_recommendations
from predictive import analyze_patient_health
from lifestyle import get_lifestyle_insights
from model import query_model
from uuid import uuid4

logger = get_logger('main')
setup_uncaught_exception_handler()

app = FastAPI(
    title="Medical Health Analytics API",
    description="AI-powered health analytics for telemedicine applications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable declaration at module level
model_interface = None

@app.on_event("startup")
async def startup_event():
    global model_interface
    logger.info("Initializing Health Analytics model...")
    from model import ModelInterface
    model_interface = await ModelInterface.get_instance()
    logger.info("Health Analytics model initialized successfully")

# API Key Security
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="API Key header missing"
        )
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key_header

class GeneticRequest(BaseModel):
    patient_id: str
    genetic_markers: List[Union[Dict[str, Any], float, int]]
    patient_info: Dict[str, Any] = {}

class PredictiveRequest(BaseModel):
    patient_id: str
    health_metrics: Dict[str, Any]
    patient_info: Dict[str, Any] = {}
    health_history: List[Dict[str, Any]] = []

class LifestyleRequest(BaseModel):
    patient_id: str
    prescription_data: List[Dict[str, Any]] = []
    patient_reports: List[str] = []
    lifestyle_data: Dict[str, Any] = {}
    medication_effectiveness: List[float] = []
    patient_info: Dict[str, Any] = {}

class MonitoringRequest(BaseModel):
    patient_id: str
    interval: int = MONITORING_INTERVAL
    enabled: bool = True

monitoring_queue = queue.Queue()
monitoring_threads = {}
monitoring_active = True

def monitor_patient(patient_id: str, interval: int):
    logger.info(f"Starting monitoring for patient {patient_id} at interval {interval}s")
    
    while monitoring_active and patient_id in monitoring_threads:
        try:
            logger.info(f"Checking health status for patient {patient_id}")
            time.sleep(interval)
        except Exception as e:
            log_exception(logger, e, f"Error monitoring patient {patient_id}")
            time.sleep(interval)

def start_monitoring(patient_id: str, interval: int = MONITORING_INTERVAL):
    if patient_id in monitoring_threads and monitoring_threads[patient_id].is_alive():
        logger.info(f"Monitoring already active for patient {patient_id}")
        return False
        
    thread = threading.Thread(
        target=monitor_patient,
        args=(patient_id, interval),
        daemon=True
    )
    monitoring_threads[patient_id] = thread
    thread.start()
    
    logger.info(f"Monitoring started for patient {patient_id}")
    return True

def stop_monitoring(patient_id: str):
    if patient_id in monitoring_threads:
        del monitoring_threads[patient_id]
        logger.info(f"Monitoring stopped for patient {patient_id}")
        return True
    return False

@app.get("/", include_in_schema=False)
async def root():
    # Redirect root to docs
    return RedirectResponse(url="/docs")

@app.post("/api/genetics/analyze")
async def analyze_genetics(request: GeneticRequest, api_key: APIKey = Depends(get_api_key)):
    try:
        logger.info(f"Received genetic analysis request for patient {request.patient_id}")
        data = request.dict()
        result = await get_personalized_recommendations(data)
        return result
    except Exception as e:
        log_exception(logger, e, "Error in genetic analysis endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictive/analyze")
async def analyze_predictive(request: PredictiveRequest, api_key: APIKey = Depends(get_api_key)):
    try:
        logger.info(f"Received predictive analysis request for patient {request.patient_id}")
        data = request.dict()
        result = await analyze_patient_health(data)
        return result
    except Exception as e:
        log_exception(logger, e, "Error in predictive analysis endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lifestyle/analyze")
async def analyze_lifestyle(request: LifestyleRequest, api_key: APIKey = Depends(get_api_key)):
    try:
        logger.info(f"Received lifestyle analysis request for patient {request.patient_id}")
        data = request.dict()
        result = await get_lifestyle_insights(data)
        return result
    except Exception as e:
        log_exception(logger, e, "Error in lifestyle analysis endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/control")
async def control_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks, api_key: APIKey = Depends(get_api_key)):
    try:
        patient_id = request.patient_id
        
        if request.enabled:
            background_tasks.add_task(start_monitoring, patient_id, request.interval)
            return {
                "status": "success",
                "message": f"Monitoring started for patient {patient_id}",
                "interval": request.interval
            }
        else:
            success = stop_monitoring(patient_id)
            return {
                "status": "success" if success else "warning",
                "message": f"Monitoring {'stopped' if success else 'was not active'} for patient {patient_id}"
            }
    except Exception as e:
        log_exception(logger, e, "Error in monitoring control endpoint")
        raise HTTPException(status_code=500, detail=str(e))


query_tasks = {}

async def run_query(task_id: str, data: Dict[str, Any], prompt_type: str):
    try:
        result = await query_model(data, prompt_type)
        query_tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        log_exception(logger, e, f"Error in run_query task {task_id}")
        query_tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/api/query")
async def query_ai_model(request: Request, background_tasks: BackgroundTasks, api_key: APIKey = Depends(get_api_key)):
    try:
        data = await request.json()
        patient_id = data.get('patient_id', 'unknown')
        prompt_type = data.get('type', 'general')
        
        # Generate a unique task ID
        task_id = str(uuid4())
        
        # Store initial task status
        query_tasks[task_id] = {"status": "processing"}
        
        # Run the query in the background
        logger.info(f"Starting background query task {task_id} for patient {patient_id}, type: {prompt_type}")
        background_tasks.add_task(run_query, task_id, data, prompt_type)
        
        # Return the task ID immediately
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        log_exception(logger, e, "Error in query_ai_model endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/query/status/{task_id}")
async def get_query_status(task_id: str, api_key: APIKey = Depends(get_api_key)):
    if task_id not in query_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return query_tasks[task_id]


@app.get("/api/status")
async def get_status(api_key: APIKey = Depends(get_api_key)):
    try:
        active_monitoring = len(monitoring_threads)
        return {
            "status": "online",
            "active_monitoring_count": active_monitoring,
            "active_patients": list(monitoring_threads.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        log_exception(logger, e, "Error in status endpoint")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/test", include_in_schema=True)
async def test_endpoint():
    return {"status": "API is working"}

@app.get("/health", include_in_schema=True)
async def health_check():
    return {"status": "online"}

def shutdown():
    global monitoring_active
    logger.info("Shutting down Medical Health Analytics API")
    monitoring_active = False
    
    for patient_id, thread in monitoring_threads.items():
        logger.info(f"Stopping monitoring for patient {patient_id}")
        if thread.is_alive():
            thread.join(timeout=1.0)

@app.on_event("shutdown")
async def shutdown_event():
    shutdown()

if __name__ == "__main__":
    try:
        logger.info("Starting Medical Health Analytics API")
        uvicorn.run(
            "main:app",
            host=API_HOST,
            port=API_PORT,
            workers=API_WORKERS
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown()
    except Exception as e:
        log_exception(logger, e, "Error starting API server")
        sys.exit(1)
