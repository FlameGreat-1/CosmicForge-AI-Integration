FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV TRANSFORMERS_CACHE /app/model_cache
ENV LOG_DIR /app/logs
ENV DATA_DIR /app/data
ENV MODEL_CACHE_DIR /app/model_cache
ENV NLTK_DATA /app/nltk_data
ENV PYTHONPATH=$APP_HOME
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV PORT=7860
ENV HOST=0.0.0.0

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install FastAPI, uvicorn and gunicorn
RUN pip install --no-cache-dir fastapi uvicorn gunicorn

# Copy all application directories
COPY medical_diagnosis_system $APP_HOME/medical_diagnosis_system
COPY cosmicforge_ai_chatbot $APP_HOME/cosmicforge_ai_chatbot
COPY CosmicForge-Health-Analytics $APP_HOME/CosmicForge-Health-Analytics
COPY Model $APP_HOME/Model

# Copy the orchestration main.py
COPY main.py $APP_HOME/main.py

# Copy the unified start.sh
COPY start.sh $APP_HOME/start.sh
RUN chmod +x $APP_HOME/start.sh

# Create __init__.py files if they don't exist
RUN touch $APP_HOME/medical_diagnosis_system/__init__.py
RUN touch $APP_HOME/cosmicforge_ai_chatbot/__init__.py
RUN touch $APP_HOME/CosmicForge-Health-Analytics/__init__.py

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Create necessary directories and set permissions
RUN mkdir -p $TRANSFORMERS_CACHE $LOG_DIR $DATA_DIR $MODEL_CACHE_DIR $NLTK_DATA \
    $APP_HOME/medical_diagnosis_system/data \
    $APP_HOME/cosmicforge_ai_chatbot/data \
    $APP_HOME/CosmicForge-Health-Analytics/data \
    && chmod 777 $TRANSFORMERS_CACHE $LOG_DIR $DATA_DIR $MODEL_CACHE_DIR $NLTK_DATA \
    $APP_HOME/medical_diagnosis_system/data \
    $APP_HOME/cosmicforge_ai_chatbot/data \
    $APP_HOME/CosmicForge-Health-Analytics/data

# Set model path environment variables to use local model
ENV MODEL_PATH="/app/Model"
ENV TRANSFORMERS_OFFLINE=1

# Set ownership of the application files
RUN chown -R nobody:nogroup $APP_HOME

# Switch to non-root user
USER nobody

# Expose ports for all three applications
EXPOSE 7860 7861 7862

# Use the start.sh script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
