#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup_log.txt"
}

# Function to handle errors
handle_error() {
    log_message "ERROR: $1"
    exit 1
}

# Set all environment variables
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export APP_HOME="/root/cosmic_app"
export TRANSFORMERS_CACHE="/root/cosmic_app/model_cache"
export LOG_DIR="/root/cosmic_app/logs"
export DATA_DIR="/root/cosmic_storage"  # Using persistent storage
export MODEL_CACHE_DIR="/root/cosmic_app/model_cache"
export NLTK_DATA="/root/cosmic_app/nltk_data"
export PYTHONPATH=$APP_HOME
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PORT=7860
export HOST=0.0.0.0

# Configuration for services
export HEALTH_PORT=${PORT:-7860}
export DIAGNOSIS_PORT=7861
export CHATBOT_PORT=7862
export WORKERS_INITIAL=${WORKERS_INITIAL:-1}
export WORKERS_FINAL=${WORKERS_FINAL:-4}
export TIMEOUT=${TIMEOUT:-300}

# Create log directory
mkdir -p "$LOG_DIR"

log_message "Starting initialization process..."
log_message "Environment variables set. APP_HOME: $APP_HOME"

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    handle_error "HF_TOKEN is not set. Please provide a valid Hugging Face token."
fi

if [ -z "$API_KEY" ]; then
    handle_error "API_KEY is not set. Please provide a valid API key."
fi

# Check Python environment
log_message "Python path: $(which python)"
log_message "Python version: $(python --version)"
log_message "Pip list:"
pip list >> "$LOG_DIR/startup_log.txt"

# Create all necessary directories
log_message "Creating necessary directories..."
mkdir -p $TRANSFORMERS_CACHE $MODEL_CACHE_DIR $NLTK_DATA \
    $APP_HOME/medical_diagnosis_system/data \
    $APP_HOME/cosmicforge_ai_chatbot/data \
    $APP_HOME/CosmicForge-Health-Analytics/data \
    $APP_HOME/logs/health_analytics \
    $APP_HOME/logs/medical_diagnosis \
    $APP_HOME/logs/medical_chatbot || handle_error "Failed to create necessary directories"
log_message "Directories created successfully."

# Set permissions (excluding DATA_DIR)
log_message "Setting permissions..."
chmod -R 755 $TRANSFORMERS_CACHE $LOG_DIR $MODEL_CACHE_DIR $NLTK_DATA \
    $APP_HOME/medical_diagnosis_system/data \
    $APP_HOME/cosmicforge_ai_chatbot/data \
    $APP_HOME/CosmicForge-Health-Analytics/data || handle_error "Failed to set permissions"
log_message "Permissions set successfully."

# Download NLTK data if not already downloaded
if [ ! -d "$NLTK_DATA/tokenizers/punkt" ]; then
    log_message "Downloading NLTK data..."
    python -m nltk.downloader -d $NLTK_DATA punkt stopwords wordnet || handle_error "Failed to download NLTK data"
    log_message "NLTK data downloaded successfully."
fi

# Create __init__.py files if they don't exist
log_message "Creating __init__.py files..."
touch $APP_HOME/medical_diagnosis_system/__init__.py
touch $APP_HOME/cosmicforge_ai_chatbot/__init__.py
touch $APP_HOME/CosmicForge-Health-Analytics/__init__.py
log_message "__init__.py files created successfully."

# Download model if it doesn't exist
if [ ! -d "$APP_HOME/Model" ] || [ -z "$(ls -A $APP_HOME/Model 2>/dev/null)" ]; then
    log_message "Model not found. Starting download from Hugging Face..."
    mkdir -p $APP_HOME/Model
    
    # Create a temporary directory for download
    TEMP_DIR="/tmp/model_download"
    mkdir -p "$TEMP_DIR"
    
    # Install huggingface_hub if not already installed
    if ! pip list | grep -q huggingface_hub; then
        log_message "Installing huggingface_hub..."
        pip install --no-cache-dir huggingface_hub || handle_error "Failed to install huggingface_hub"
    fi
    
    # Use Hugging Face CLI to download with retry logic
    log_message "Starting model download process..."
    python -c "
import sys
import time
from huggingface_hub import snapshot_download
import os
import shutil

max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        print(f'Download attempt {attempt + 1}/{max_retries}...')
        model_path = snapshot_download(
            repo_id='FlameGreat01/Medical_Diagnosis_System',
            token=os.environ.get('HF_TOKEN'),
            local_dir='$TEMP_DIR',
            local_dir_use_symlinks=False,
            ignore_patterns=[
                'README.md', 
                'USE_POLICY.md',
                '.gitattributes',
                '.git*',
                '*.lock',
                '*.metadata',
                'LICENSE.txt'
            ],
            resume_download=True
        )
        print(f'Model downloaded successfully to {model_path}')
        
        print(f'Moving files to {os.environ.get(\"APP_HOME\")}/Model...')
        for root, dirs, files in os.walk(model_path):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, model_path)
                dst_path = os.path.join(os.environ.get('APP_HOME') + '/Model', rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f'Moved: {rel_path}')
        
        sys.exit(0)  # Success
    except Exception as e:
        print(f'Attempt {attempt + 1} failed: {str(e)}')
        if attempt < max_retries - 1:
            print(f'Retrying in {retry_delay} seconds...')
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        else:
            print('All download attempts failed.')
            sys.exit(1)  # Failure
" || handle_error "Download script failed"
    
    # Clean up temp directory
    rm -rf "$TEMP_DIR"
    
    # Verify model files exist
    if [ -z "$(ls -A $APP_HOME/Model 2>/dev/null)" ]; then
        handle_error "Model directory is empty after download"
    fi
    
    # Set permissions
    chmod -R 755 $APP_HOME/Model
    log_message "Model download complete."
    
    # Log model directory contents
    log_message "Contents of $APP_HOME/Model:"
    ls -R $APP_HOME/Model >> "$LOG_DIR/startup_log.txt"
else
    log_message "Model already exists. Skipping download."
fi

# Ensure all dependencies are installed
if [ -f "$APP_HOME/requirements.txt" ]; then
    log_message "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r "$APP_HOME/requirements.txt" || handle_error "Failed to install dependencies"
    log_message "Dependencies installed successfully."
else
    handle_error "requirements.txt not found in $APP_HOME. Cannot proceed without dependencies."
fi

# Set model environment variables
export MODEL_PATH="$APP_HOME/Model"
export TRANSFORMERS_OFFLINE=1

log_message "Starting AI Integration Platform using orchestration main.py..."
log_message "Using model at $MODEL_PATH"

# Handle termination signals
trap "log_message 'Shutting down all services...'; exit" SIGINT SIGTERM

# Check if main.py exists
if [ ! -f "$APP_HOME/main.py" ]; then
    handle_error "main.py not found in $APP_HOME"
fi

# Run the orchestration main.py with timeout
log_message "Executing main.py..."
timeout 3600 python main.py 2>&1 | tee -a "$LOG_DIR/main_execution.log" || handle_error "main.py execution failed or timed out after 1 hour"

# Check if main.py exited successfully
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    handle_error "main.py exited with non-zero status"
fi

log_message "Startup script completed successfully."
