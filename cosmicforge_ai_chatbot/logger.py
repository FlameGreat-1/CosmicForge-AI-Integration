import logging
from .config import Config

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(Config.LOG_LEVEL))

    formatter = logging.Formatter(Config.LOG_FORMAT)

    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not Config.is_production():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent double logging in jupyter notebooks

    return logger
