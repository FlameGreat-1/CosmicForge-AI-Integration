import os
import sys
import logging
import subprocess
import threading
import time
import signal
import requests
import psutil
from pathlib import Path

# Use APP_HOME from environment variable (set in start.sh)
APP_HOME = os.environ.get("APP_HOME", "/root/cosmic_app")

# Set up logging
log_dir = Path(f"{APP_HOME}/logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "orchestrator.log")
    ]
)
logger = logging.getLogger("ai_integration_orchestrator")

# Configuration
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", 7860))
DIAGNOSIS_PORT = int(os.environ.get("DIAGNOSIS_PORT", 7861))
CHATBOT_PORT = int(os.environ.get("CHATBOT_PORT", 7862))
HOST = os.environ.get("HOST", "0.0.0.0")
WORKERS_INITIAL = int(os.environ.get("WORKERS_INITIAL", 1))
WORKERS_FINAL = int(os.environ.get("WORKERS_FINAL", 4))
TIMEOUT = int(os.environ.get("TIMEOUT", 300))
MODEL_PATH = os.environ.get("MODEL_PATH", f"{APP_HOME}/Model")

# Global process tracking
processes = {
    "health_analytics": None,
    "medical_diagnosis": None,
    "medical_chatbot": None
}

# Create log directories
for app in ["health_analytics", "medical_diagnosis", "medical_chatbot"]:
    app_log_dir = log_dir / app
    app_log_dir.mkdir(exist_ok=True)

def check_health_model_loaded():
    """Check if Health Analytics model is loaded"""
    logger.info("Checking if Health Analytics model is loaded...")
    
    for i in range(30):
        try:
            logger.info(f"Health check attempt {i+1}/30...")
            response = requests.get(f"http://{HOST}:{HEALTH_PORT}/health", timeout=5)
            if response.status_code == 200 and "online" in response.text:
                logger.info("Health Analytics model loaded successfully!")
                return True
        except requests.RequestException as e:
            logger.warning(f"Health check failed: {str(e)}")
        
        time.sleep(10)
    
    logger.error("Timed out waiting for Health Analytics model to load")
    return False

def run_health_analytics():
    """Run the Health Analytics application with worker scaling"""
    global processes
    
    # Create log files
    access_log = log_dir / "health_analytics" / "access.log"
    error_log = log_dir / "health_analytics" / "error.log"
    
    # Start with initial workers for model loading
    logger.info(f"Starting Health Analytics on port {HEALTH_PORT} with {WORKERS_INITIAL} worker(s)...")
    
    os.chdir(f"{APP_HOME}/CosmicForge-Health-Analytics")
    
    # Use gunicorn for Health Analytics (as in original start.sh)
    cmd = [
        "gunicorn", 
        "main:app", 
        "-w", str(WORKERS_INITIAL),
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{HOST}:{HEALTH_PORT}",
        "--timeout", str(TIMEOUT),
        "--access-logfile", str(access_log),
        "--error-logfile", str(error_log)
    ]
    
    processes["health_analytics"] = subprocess.Popen(cmd)
    
    # Wait for model to load
    model_loaded = check_health_model_loaded()
    
    # If model loaded successfully and we want more workers, restart
    if model_loaded and WORKERS_FINAL > WORKERS_INITIAL:
        logger.info(f"Restarting Health Analytics with {WORKERS_FINAL} workers...")
        
        # Terminate the initial process
        if processes["health_analytics"]:
            processes["health_analytics"].terminate()
            processes["health_analytics"].wait()
        
        # Start with final worker count
        cmd = [
            "gunicorn", 
            "main:app", 
            "-w", str(WORKERS_FINAL),
            "-k", "uvicorn.workers.UvicornWorker",
            "--bind", f"{HOST}:{HEALTH_PORT}",
            "--timeout", str(TIMEOUT),
            "--preload",
            "--access-logfile", str(access_log),
            "--error-logfile", str(error_log)
        ]
        
        processes["health_analytics"] = subprocess.Popen(cmd)

def run_medical_diagnosis():
    """Run the Medical Diagnosis application"""
    global processes
    
    # Create log file
    log_file = log_dir / "medical_diagnosis" / "access.log"
    
    logger.info(f"Starting Medical Diagnosis System on port {DIAGNOSIS_PORT}...")
    
    os.chdir(f"{APP_HOME}/medical_diagnosis_system")
    
    # Use uvicorn for Medical Diagnosis (as in original Dockerfile)
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", HOST, 
        "--port", str(DIAGNOSIS_PORT),
        "--log-level", "info",
        "--log-file", str(log_file)
    ]
    
    processes["medical_diagnosis"] = subprocess.Popen(cmd)

def run_medical_chatbot():
    """Run the Medical Chatbot application"""
    global processes
    
    # Create log file
    log_file = log_dir / "medical_chatbot" / "access.log"
    
    logger.info(f"Starting Medical Chatbot on port {CHATBOT_PORT}...")
    
    os.chdir(f"{APP_HOME}/cosmicforge_ai_chatbot")
    
    # Use uvicorn for Medical Chatbot (as in original Dockerfile)
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", HOST, 
        "--port", str(CHATBOT_PORT),
        "--log-level", "info",
        "--log-file", str(log_file)
    ]
    
    processes["medical_chatbot"] = subprocess.Popen(cmd)

def monitor_processes():
    """Monitor all processes and restart if they fail"""
    global processes
    
    while True:
        # Check Health Analytics
        if processes["health_analytics"] and processes["health_analytics"].poll() is not None:
            logger.warning("Health Analytics process died, restarting...")
            health_thread = threading.Thread(target=run_health_analytics)
            health_thread.daemon = True
            health_thread.start()
        
        # Check Medical Diagnosis
        if processes["medical_diagnosis"] and processes["medical_diagnosis"].poll() is not None:
            logger.warning("Medical Diagnosis process died, restarting...")
            diagnosis_thread = threading.Thread(target=run_medical_diagnosis)
            diagnosis_thread.daemon = True
            diagnosis_thread.start()
        
        # Check Medical Chatbot
        if processes["medical_chatbot"] and processes["medical_chatbot"].poll() is not None:
            logger.warning("Medical Chatbot process died, restarting...")
            chatbot_thread = threading.Thread(target=run_medical_chatbot)
            chatbot_thread.daemon = True
            chatbot_thread.start()
        
        # Log resource usage
        log_resource_usage()
        
        # Sleep before next check
        time.sleep(30)

def log_resource_usage():
    """Log system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        logger.info(f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
    except Exception as e:
        logger.error(f"Error logging resource usage: {str(e)}")

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down all services...")
    
    # Terminate all processes
    for name, process in processes.items():
        if process:
            logger.info(f"Terminating {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} did not terminate gracefully, killing...")
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating {name}: {str(e)}")
    
    logger.info("All services shut down")
    sys.exit(0)

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please ensure the model is downloaded.")
        sys.exit(1)
        
    # Print startup banner
    logger.info("=" * 80)
    logger.info("Starting AI Integration Platform")
    logger.info(f"Using local model at {MODEL_PATH}")
    logger.info(f"Health Analytics: http://{HOST}:{HEALTH_PORT}")
    logger.info(f"Medical Diagnosis: http://{HOST}:{DIAGNOSIS_PORT}")
    logger.info(f"Medical Chatbot: http://{HOST}:{CHATBOT_PORT}")
    logger.info("=" * 80)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    # Create threads for each application
    health_thread = threading.Thread(target=run_health_analytics)
    diagnosis_thread = threading.Thread(target=run_medical_diagnosis)
    chatbot_thread = threading.Thread(target=run_medical_chatbot)
    
    # Set as daemon threads so they exit when main thread exits
    health_thread.daemon = True
    diagnosis_thread.daemon = True
    chatbot_thread.daemon = True
    
    # Start all threads
    health_thread.start()
    diagnosis_thread.start()
    chatbot_thread.start()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_processes)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for all threads to complete (which won't happen unless interrupted)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        handle_shutdown(signal.SIGINT, None)
