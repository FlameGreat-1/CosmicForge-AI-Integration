# CosmicForge Medical AI Platform

<div align="center">
  
![CosmicForge Logo](https://via.placeholder.com/150?text=CosmicForge)

**Advanced AI Solutions for Medical Analysis, Diagnosis, and Information**

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [Deployment on RunPod](#deployment-on-runpod)
  - [Prerequisites](#prerequisites)
  - [Quick Deployment](#quick-deployment)
  - [Accessing Services](#accessing-services)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
  - [Medical Diagnosis System API](#1-medical-diagnosis-system-api-port-7861)
  - [CosmicForge AI Medical Chatbot API](#2-cosmicforge-ai-medical-chatbot-api-port-7862)
  - [Health Analytics API](#3-health-analytics-api-port-7860)
- [Performance Considerations](#performance-considerations)
- [Monitoring and Logs](#monitoring-and-logs)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

## Overview

CosmicForge Medical AI Platform is a comprehensive suite of integrated AI applications designed for advanced medical analysis, diagnosis, and information. The platform is optimized for deployment on RunPod, providing high-performance GPU acceleration for medical AI applications.

## Key Components

The platform consists of three main components:

| Component | Description | Port |
|-----------|-------------|------|
| **Medical Diagnosis System** | AI-powered symptom analysis and disease prediction | 7861 |
| **CosmicForge AI Medical Chatbot** | Conversational AI for medical information and guidance | 7862 |
| **Health Analytics** | Advanced health metrics analysis and personalized recommendations | 7860 |

## Deployment on RunPod

### Prerequisites

- RunPod account with GPU access
- Hugging Face account and access token
- Model hosted on Hugging Face at `FlameGreat01/Medical_Diagnosis_System`

### Quick Deployment

1. **Create a new RunPod pod with the following settings:**
   - Select a GPU template (recommended: at least 24GB VRAM)
   - Set volume size to at least 25GB
   - Add the following environment variables:
     ```
     HF_TOKEN=your_hugging_face_token
     API_KEY=your_secure_api_key
     ```

2. **Deploy using the GitHub repository:**
   ```bash
   git clone https://github.com/your-username/cosmicforge-medical-platform.git
   cd cosmicforge-medical-platform
   docker build -t cosmicforge-medical-platform .
   docker run -p 7860:7860 -p 7861:7861 -p 7862:7862 --gpus all cosmicforge-medical-platform
   ```

### Accessing Services

After successful deployment, you can access the services at:

- **Medical Diagnosis System**: `http://your-runpod-ip:7861`
- **CosmicForge AI Medical Chatbot**: `http://your-runpod-ip:7862`
- **Health Analytics**: `http://your-runpod-ip:7860`

## Architecture

The platform runs three separate services that share a common AI model:

```
┌─────────────────────────────────────┐
│         RunPod Environment          │
│                                     │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Medical   │  │ CosmicForge │   │
│  │  Diagnosis  │  │    Medical  │   │
│  │   System    │  │   Chatbot   │   │
│  │  (Port 7861)│  │ (Port 7862) │   │
│  └─────────────┘  └─────────────┘   │
│         │               │           │
│         ▼               ▼           │
│  ┌─────────────────────────────┐    │
│  │        Shared Model         │    │
│  │  (/app/Model directory)     │    │
│  └─────────────────────────────┘    │
│         ▲                           │
│         │                           │
│  ┌─────────────┐                    │
│  │   Health    │                    │
│  │  Analytics  │                    │
│  │(Port 7860)  │                    │
│  └─────────────┘                    │
└─────────────────────────────────────┘
```

## Project Structure

```
/app
├── Model/                              # Model directory (downloaded from Hugging Face)
│   ├── config.json                     # Model configuration
│   ├── generation_config.json          # Generation parameters
│   ├── model-00001-of-00002.safetensors # Model weights part 1
│   ├── model-00002-of-00002.safetensors # Model weights part 2
│   ├── model.safetensors.index.json    # Model index file
│   ├── orig_params.json                # Original parameters
│   ├── params.json                     # Model parameters
│   ├── special_tokens_map.json         # Special tokens mapping
│   ├── tokenizer.json                  # Tokenizer configuration
│   ├── tokenizer.model                 # Tokenizer model
│   └── tokenizer_config.json           # Tokenizer configuration
│
├── medical_diagnosis_system/           # Medical Diagnosis System application
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Configuration settings
│   ├── data/                           # Data storage directory
│   ├── logger.py                       # Logging configuration
│   ├── logs/                           # Log files directory
│   ├── main.py                         # FastAPI application entry point
│   ├── medical_diagnosis.py            # Diagnosis logic
│   ├── model.py                        # Model interface
│   └── models/                         # Model-related files
│
├── cosmicforge_ai_chatbot/             # Medical Chatbot application
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Configuration settings
│   ├── cosmicforge_chatbot.py          # Chatbot logic
│   ├── data/                           # Data storage directory
│   ├── logger.py                       # Logging configuration
│   ├── logs/                           # Log files directory
│   ├── main.py                         # FastAPI application entry point
│   ├── model.py                        # Model interface
│   └── models/                         # Model-related files
│
├── CosmicForge-Health-Analytics/       # Health Analytics application
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Configuration settings
│   ├── data/                           # Data storage directory
│   ├── genetics.py                     # Genetics analysis logic
│   ├── lifestyle.py                    # Lifestyle analysis logic
│   ├── logger.py                       # Logging configuration
│   ├── main.py                         # FastAPI application entry point
│   ├── model.py                        # Model interface
│   └── predictive.py                   # Predictive analysis logic
│
├── logs/                               # Root logs directory
│   ├── health_analytics/               # Health Analytics logs
│   ├── medical_diagnosis/              # Medical Diagnosis logs
│   ├── medical_chatbot/                # Medical Chatbot logs
│   └── orchestrator.log                # Main orchestrator log
│
├── Dockerfile                          # Docker configuration
├── README.md                           # Project documentation
├── main.py                             # Orchestration script
├── requirements.txt                    # Python dependencies
└── start.sh                            # Startup script
```

## API Documentation

### 1. Medical Diagnosis System API (Port 7861)

**Authentication**  
Use the API key in the 'access_token' header for all requests.

**Main Endpoints**
- `POST /api/diagnose`: Submit symptoms for diagnosis
- `GET /api/diagnosis/status/{task_id}`: Check diagnosis status
- `GET /api/diagnosis/{diagnosis_id}`: Retrieve completed diagnosis
- `GET /health`: Check API health

**Example Request**
```bash
curl -X POST "http://your-runpod-ip:7861/api/diagnose" \
  -H "Content-Type: application/json" \
  -H "access_token: your_api_key" \
  -d '{"symptoms": ["fever", "cough", "fatigue"], "patient_info": {"age": 45, "gender": "male"}}'
```

### 2. CosmicForge AI Medical Chatbot API (Port 7862)

**Authentication**  
Use the API key in the 'access_token' header for all requests.

**Main Endpoints**
- `POST /api/chat`: Submit a medical query
- `GET /api/chat/status/{task_id}`: Check chat response status
- `GET /api/chat/{response_id}`: Retrieve completed chat response
- `GET /health`: Check API health

**Example Request**
```bash
curl -X POST "http://your-runpod-ip:7862/api/chat" \
  -H "Content-Type: application/json" \
  -H "access_token: your_api_key" \
  -d '{"message": "What are the symptoms of diabetes?"}'
```

### 3. Health Analytics API (Port 7860)

**Authentication**  
Use the API key in the 'access_token' header for all requests.

**Main Endpoints**
- `POST /api/genetics/analyze`: Analyze genetic markers
- `POST /api/predictive/analyze`: Analyze health metrics
- `POST /api/lifestyle/analyze`: Analyze lifestyle data
- `POST /api/query`: Submit a general health query
- `GET /api/query/status/{task_id}`: Check query status
- `GET /health`: Check API health

**Example Request**
```bash
curl -X POST "http://your-runpod-ip:7860/api/predictive/analyze" \
  -H "Content-Type: application/json" \
  -H "access_token: your_api_key" \
  -d '{"patient_id": "P12345", "health_metrics": {"blood_pressure": "120/80", "heart_rate": 72}}'
```

## Performance Considerations

- **Memory Usage**: The model requires approximately 7GB of VRAM
- **Startup Time**: Initial startup may take 5-10 minutes as the model is downloaded from Hugging Face
- **Concurrent Requests**: The system can handle multiple concurrent requests, but performance depends on available GPU resources

## Monitoring and Logs

Logs are stored in the following directories:
- `/app/logs/health_analytics/`
- `/app/logs/medical_diagnosis/`
- `/app/logs/medical_chatbot/`

To view logs in real-time:
```bash
docker exec -it container_id tail -f /app/logs/orchestrator.log
```

## Troubleshooting

### Common Issues

| Issue | Resolution |
|-------|------------|
| **Model Download Failure** | - Check your HF_TOKEN is valid<br>- Ensure RunPod has internet access<br>- Check disk space availability |
| **Service Not Starting** | - Check logs for specific error messages<br>- Ensure GPU is properly detected<br>- Verify port mappings are correct |
| **Out of Memory Errors** | - Choose a RunPod instance with more VRAM<br>- Reduce WORKERS_FINAL environment variable |

## Security Notes

- Always use a strong, unique API_KEY
- The platform is designed for internal or authenticated use only
- No PHI (Protected Health Information) should be stored permanently

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and Hugging Face Transformers
- Model based on state-of-the-art medical language models
- Optimized for GPU acceleration on RunPod

## Support

For support or questions, please open an issue on the GitHub repository or contact support@example.com.





Verify that the repository has been cloned:

ls /root/cosmic_app
Insert at cursor



If the repository is not there, clone it manually:

git clone https://github.com/FlameGreat-1/CosmicForge-AI-Integration.git /root/cosmic_app
Insert at cursor



Check if the start.sh script exists:

ls /root/cosmic_app/start.sh


Set the required environment variables:

export HF_TOKEN=
export API_KEY=LkGaxbtyLthjp0VUA9TgUvnMY0aweonr
Insert at cursor



Run the start script again:

bash /root/cosmic_app/start.sh
Insert at cursor



After the script completes, check if the processes are running:

ps aux | grep python
Insert at cursor



Check the logs:

cat /root/cosmic_app/logs/startup_log.txt
cat /root/cosmic_app/logs/main_execution.log
Insert at cursor



If netstat is not available, you can use this alternative to check if the ports are in use:

ss -tulpn | grep -E '7860|7861|786




To resolve the NLTK installation issue and complete the setup, follow these steps:


Install NLTK:

pip install nltk
Insert at cursor



Modify the start.sh script to include NLTK installation:
Add this line before the NLTK data download command:

pip install --no-cache-dir nltk
Insert at cursor



Run the start script again:

bash /root/cosmic_app/start.sh
Insert at cursor



If the script completes successfully, check if the processes are running:

ps aux | grep python
Insert at cursor



Check the logs:

cat /root/cosmic_app/logs/startup_log.txt
cat /root/cosmic_app/logs/main_execution.log
Insert at cursor



Check if the ports are in use:

ss -tulpn | grep -E '7860|7861|7862
