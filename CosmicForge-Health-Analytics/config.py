import os
from typing import Dict, Any
import json

# Base configuration
BASE_CONFIG = {
    # Model settings
    "MODEL_PATH": os.environ.get("MODEL_PATH", "/root/cosmic_app/Model"),
    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1",
    
    # API settings
    "API_HOST": os.environ.get("HOST", "0.0.0.0"),
    "API_PORT": int(os.environ.get("HEALTH_PORT", "7860")),
    "API_WORKERS": int(os.environ.get("WORKERS_FINAL", "4")),
    "API_KEY": os.environ.get("API_KEY"),
    
    # Performance settings
    "BATCH_SIZE": 1,  # Process one patient at a time for real-time analysis
    "MAX_QUEUE_SIZE": 100,  # Maximum number of requests to queue
    "WORKER_THREADS": 4,  # Number of worker threads for processing
    
    # Monitoring settings
    "MONITORING_INTERVAL": 300,  # Check patient data every 5 minutes (in seconds)
    "CRITICAL_ALERT_THRESHOLD": 0.8,  # Threshold for critical health alerts
    
    # Request handling
    "REQUEST_TIMEOUT": 30,  # Timeout for model requests in seconds
    "MAX_RETRIES": 3,  # Maximum number of retries for failed requests
    "RETRY_DELAY": 2,  # Initial delay between retries in seconds
    
    # Data processing
    "GENETIC_DATA_THRESHOLD": 0.7,  # Threshold for genetic risk factors
    "PREDICTION_CONFIDENCE_THRESHOLD": 0.75,  # Minimum confidence for predictions
    
    # Paths - Aligning with start.sh and medical_diagnosis_system/config.py
    "APP_HOME": os.environ.get('APP_HOME', '/root/cosmic_app'),
    "BASE_DIR": os.environ.get('BASE_DIR', '/root/cosmic_app/CosmicForge-Health-Analytics'),
    "MODEL_CACHE_DIR": os.environ.get("MODEL_CACHE_DIR", "/root/cosmic_app/model_cache"),
    "LOG_DIR": os.environ.get("LOG_DIR", "/root/cosmic_app/logs/health_analytics"),
    "DATA_DIR": os.environ.get("DATA_DIR", "/root/cosmic_storage"),
    "NLTK_DATA_PATH": os.environ.get("NLTK_DATA", "/root/cosmic_app/nltk_data"),
    
    # Logging
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE_MAX_BYTES": 10485760,  # 10MB
    "LOG_BACKUP_COUNT": 5,
}

# Create necessary directories - Wrapped in try-except to handle any permission issues
try:
    for directory in [BASE_CONFIG["LOG_DIR"], 
                     os.path.join(BASE_CONFIG["BASE_DIR"], "data"), 
                     BASE_CONFIG["MODEL_CACHE_DIR"], 
                     BASE_CONFIG["NLTK_DATA_PATH"]]:
        os.makedirs(directory, exist_ok=True)
except PermissionError:
    print(f"Warning: Permission denied when creating directories. Using fallback to /tmp")
    # Fallback to /tmp if there are permission issues
    BASE_CONFIG["MODEL_CACHE_DIR"] = "/tmp/model_cache"
    BASE_CONFIG["LOG_DIR"] = "/tmp/logs/health_analytics"
    BASE_CONFIG["DATA_DIR"] = "/tmp/data"
    BASE_CONFIG["NLTK_DATA_PATH"] = "/tmp/nltk_data"
    
    # Try again with /tmp paths
    for directory in [BASE_CONFIG["LOG_DIR"], 
                     os.path.join("/tmp/data", "health_analytics"), 
                     BASE_CONFIG["MODEL_CACHE_DIR"], 
                     BASE_CONFIG["NLTK_DATA_PATH"]]:
        os.makedirs(directory, exist_ok=True)

# Environment-specific configuration
ENV = os.environ.get("ENVIRONMENT", "production")

# Load environment-specific config if available
env_config_path = os.path.join(BASE_CONFIG["BASE_DIR"], f"config_{ENV}.json")
if os.path.exists(env_config_path):
    try:
        with open(env_config_path, 'r') as f:
            env_config = json.load(f)
            BASE_CONFIG.update(env_config)
    except Exception as e:
        print(f"Warning: Could not load environment config from {env_config_path}: {str(e)}")

# Export all config variables to the module namespace
for key, value in BASE_CONFIG.items():
    globals()[key] = value

def get_config() -> Dict[str, Any]:
    """Get the current configuration as a dictionary."""
    return {k: v for k, v in BASE_CONFIG.items()}

def update_config(new_config: Dict[str, Any]) -> None:
    """
    Update configuration values.
    
    Args:
        new_config: Dictionary containing new configuration values
    """
    BASE_CONFIG.update(new_config)
    
    # Update module namespace
    for key, value in new_config.items():
        globals()[key] = value

def is_production():
    """Check if the environment is production."""
    return ENV.lower() == 'production'

def create_directories():
    """Create all necessary directories."""
    directories = [
        LOG_DIR,
        os.path.join(BASE_DIR, "data"),
        MODEL_CACHE_DIR,
        NLTK_DATA_PATH
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)