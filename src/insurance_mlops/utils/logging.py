import logging
import sys
from typing import Optional

def setup_logger(name: str, level: str = "INFO", log_format: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
