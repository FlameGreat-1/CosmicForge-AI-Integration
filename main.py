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

def check_service_health(port, endpoint="/health"):
    """Check if a service is healthy"""
    for i in range(30):
        try:
            response = requests.get(f"http://{HOST}:{port}{endpoint}", timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(10)
    return False

def run_subprocess(cmd, name):
    """Run a subprocess with proper logging and error handling"""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes[name] = process
        
        def log_output(stream, log_level):
            for line in stream:
                logger.log(log_level, f"{name}: {line.decode().strip()}")
        
        threading.Thread(target=log_output, args=(process.stdout, logging.INFO), daemon=True).start()
        threading.Thread(target=log_output, args=(process.stderr, logging.ERROR), daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Failed to start {name}: {str(e)}")
        return None

def run_health_analytics():
    """Run the Health Analytics application with worker scaling"""
    access_log = log_dir / "health_analytics" / "access.log"
    error_log = log_dir / "health_analytics" / "error.log"
    
    os.chdir(f"{APP_HOME}/CosmicForge-Health-Analytics")
    
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
    
    process = run_subprocess(cmd, "health_analytics")
    
    if process and check_health_model_loaded():
        if WORKERS_FINAL > WORKERS_INITIAL:
            logger.info(f"Restarting Health Analytics with {WORKERS_FINAL} workers...")
            process.terminate()
            process.wait()
            
            cmd[3] = str(WORKERS_FINAL)
            cmd.extend(["--preload"])
            
            run_subprocess(cmd, "health_analytics")
    else:
        logger.error("Failed to start Health Analytics")

def run_medical_diagnosis():
    """Run the Medical Diagnosis application"""
    log_file = log_dir / "medical_diagnosis" / "access.log"
    
    os.chdir(f"{APP_HOME}/medical_diagnosis_system")
    
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", HOST, 
        "--port", str(DIAGNOSIS_PORT),
        "--log-level", "info",
        "--log-file", str(log_file)
    ]
    
    run_subprocess(cmd, "medical_diagnosis")

def run_medical_chatbot():
    """Run the Medical Chatbot application"""
    log_file = log_dir / "medical_chatbot" / "access.log"
    
    os.chdir(f"{APP_HOME}/cosmicforge_ai_chatbot")
    
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", HOST, 
        "--port", str(CHATBOT_PORT),
        "--log-level", "info",
        "--log-file", str(log_file)
    ]
    
    run_subprocess(cmd, "medical_chatbot")

def monitor_processes():
    """Monitor all processes and restart if they fail"""
    while True:
        for name, process in processes.items():
            if process and process.poll() is not None:
                logger.warning(f"{name} process died, restarting...")
                if name == "health_analytics":
                    threading.Thread(target=run_health_analytics, daemon=True).start()
                elif name == "medical_diagnosis":
                    threading.Thread(target=run_medical_diagnosis, daemon=True).start()
                elif name == "medical_chatbot":
                    threading.Thread(target=run_medical_chatbot, daemon=True).start()
        
        log_resource_usage()
        time.sleep(30)

def log_resource_usage():
    """Log system and process resource usage"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        logger.info(f"System resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
        
        for name, process in processes.items():
            if process:
                try:
                    p = psutil.Process(process.pid)
                    logger.info(f"{name} resource usage - CPU: {p.cpu_percent()}%, Memory: {p.memory_percent()}%")
                except psutil.NoSuchProcess:
                    logger.warning(f"{name} process not found")
    except Exception as e:
        logger.error(f"Error logging resource usage: {str(e)}")

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down all services...")
    
    for name, process in processes.items():
        if process:
            logger.info(f"Terminating {name}...")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"{name} did not terminate gracefully, killing...")
                process.kill()
            except Exception as e:
                logger.error(f"Error terminating {name}: {str(e)}")
    
    logger.info("All services shut down")
    sys.exit(0)

def validate_environment():
    """Validate critical environment variables"""
    required_vars = ["APP_HOME", "MODEL_PATH", "HF_TOKEN", "API_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Required environment variable {var} is not set")
            sys.exit(1)

if __name__ == "__main__":
    validate_environment()
    
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Please ensure the model is downloaded.")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Starting AI Integration Platform")
    logger.info(f"Using local model at {MODEL_PATH}")
    logger.info(f"Health Analytics: http://{HOST}:{HEALTH_PORT}")
    logger.info(f"Medical Diagnosis: http://{HOST}:{DIAGNOSIS_PORT}")
    logger.info(f"Medical Chatbot: http://{HOST}:{CHATBOT_PORT}")
    logger.info("=" * 80)
    
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    threading.Thread(target=run_health_analytics, daemon=True).start()
    threading.Thread(target=run_medical_diagnosis, daemon=True).start()
    threading.Thread(target=run_medical_chatbot, daemon=True).start()
    
    monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
    monitor_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        handle_shutdown(signal.SIGINT, None)
