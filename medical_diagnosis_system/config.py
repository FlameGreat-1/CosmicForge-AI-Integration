import os

class Config:
    # Base directories
    APP_HOME = os.environ.get('APP_HOME', '/root/cosmic_app')
    BASE_DIR = os.path.join(APP_HOME, 'medical_diagnosis_system')

    # Model configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(APP_HOME, 'Model'))
    MODEL_VERSION = '1.0'

    # Data and logging directories
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_DIR = os.environ.get('LOG_DIR', os.path.join(APP_HOME, 'logs/medical_diagnosis'))
    PDF_DIR = os.path.join(LOG_DIR, 'pdf_reports')
    LOG_FILE = os.path.join(LOG_DIR, 'medical_diagnosis.log')

    # API configuration
    API_HOST = os.environ.get('HOST', '0.0.0.0')
    API_PORT = int(os.environ.get('DIAGNOSIS_PORT', 7861))

    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Offline mode for transformers
    TRANSFORMERS_OFFLINE = os.environ.get('TRANSFORMERS_OFFLINE', '1') == '1'

    @classmethod
    def is_production(cls):
        return os.environ.get('ENVIRONMENT', 'production').lower() == 'production'

    @classmethod
    def create_directories(cls):
        for directory in [cls.DATA_DIR, cls.LOG_DIR, cls.PDF_DIR]:
            os.makedirs(directory, exist_ok=True)
