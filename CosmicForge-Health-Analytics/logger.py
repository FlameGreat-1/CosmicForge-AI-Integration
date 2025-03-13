import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import traceback
from typing import Optional

# Import configuration
from config import LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_FILE_MAX_BYTES, LOG_BACKUP_COUNT

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Map string log levels to logging constants
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger, typically the module name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        # Set log level
        level = LOG_LEVELS.get(LOG_LEVEL, logging.INFO)
        logger.setLevel(level)
        
        # Create log file name with date
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(LOG_DIR, f"{name}_{date_str}.log")
        
        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Main application logger
app_logger = get_logger('medical_analytics')

def log_exception(logger: logging.Logger, e: Exception, context: str = "") -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        e: Exception to log
        context: Additional context information
    """
    error_msg = f"{context}: {str(e)}" if context else str(e)
    logger.error(error_msg)
    logger.error(traceback.format_exc())

def setup_uncaught_exception_handler() -> None:
    """Setup global exception handler for uncaught exceptions."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        app_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    app_logger.info("Global exception handler configured")
