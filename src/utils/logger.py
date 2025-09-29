import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOGS_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Create logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # logs to console
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
