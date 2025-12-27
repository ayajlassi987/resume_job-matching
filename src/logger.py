"""Structured logging utilities."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config import LOG_LEVEL, LOG_FILE


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logger(
    name: str = "interview_copilot",
    level: str = None,
    log_file: str = None,
    use_json: bool = True
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        use_json: Use JSON formatting
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level or LOG_LEVEL, logging.INFO))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    logger.addHandler(console_handler)
    
    # File handler
    if log_file or LOG_FILE:
        log_path = Path(log_file or LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def log_request(method: str, endpoint: str, duration: float, status_code: int = 200):
    """Log API request."""
    logger.info(
        "API request",
        extra={
            "extra_fields": {
                "type": "api_request",
                "method": method,
                "endpoint": endpoint,
                "duration_ms": duration * 1000,
                "status_code": status_code
            }
        }
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    logger.error(
        f"Error: {str(error)}",
        exc_info=error,
        extra={
            "extra_fields": {
                "type": "error",
                "context": context or {}
            }
        }
    )


def log_performance(operation: str, duration: float, metadata: Dict[str, Any] = None):
    """Log performance metrics."""
    logger.info(
        f"Performance: {operation}",
        extra={
            "extra_fields": {
                "type": "performance",
                "operation": operation,
                "duration_ms": duration * 1000,
                "metadata": metadata or {}
            }
        }
    )

