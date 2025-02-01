import logging
import colorlog
import os
from pathlib import Path

# Dynamically find the root project directory
current_dir = Path(__file__).resolve()

# Traverse up until the directory name matches the project folder
MAIN_DIR = current_dir
while MAIN_DIR.name != "Chatbot-For-Dental-Clinic":
    MAIN_DIR = MAIN_DIR.parent

# Define the log directory
LOG_DIR = MAIN_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

# Formatter for file logs (clean, no colors)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Custom emoji mapping per logger
LOG_EMOJIS = {
    "HighLevelErrors": "🔴",
    "ModelingOperation": "🟣",
    "APIOperation": "🟢",
    "DataOperation": "🔵",
    "PipelineOperation": "🟡"
}

# Custom color mapping per logger
LOG_COLORS = {
    "HighLevelErrors": "red",
    "ModelingOperation": "cyan",
    "APIOperation": "green",
    "DataOperation": "blue",
    "PipelineOperation": "yellow"
}

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with emoji-based color formatting for console and clean file logs."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.hasHandlers():
        # File handler (plain logs)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)

        # Console handler (emoji + color logs)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = colorlog.ColoredFormatter(
            f"%(asctime)s - %(name)s - {LOG_EMOJIS.get(name, '⚪')}%(log_color)s%(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={name: LOG_COLORS.get(name, "white")}  # Default to white if no color is assigned
        )
        console_handler.setFormatter(console_formatter)

        # Attach handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Initialize loggers with unique colors & emojis
HighLevelErrors = setup_logger("HighLevelErrors", LOG_DIR / "HighLevelErrors.log", logging.ERROR)
ModelingOperation = setup_logger("ModelingOperation", LOG_DIR / "ModelingOperation.log", logging.INFO)
APIOperation = setup_logger("APIOperation", LOG_DIR / "APIOperation.log", logging.INFO)
DataOperation = setup_logger("DataOperation", LOG_DIR / "DataOperation.log", logging.INFO)
PipelineOperation = setup_logger("PipelineOperation", LOG_DIR / "PipelineOperation.log", logging.INFO)

if __name__ == "__main__":
    # Sample logs with custom colors and emojis
    HighLevelErrors.error("Critical system failure detected!")  # 🔴
    ModelingOperation.info("Modeling pipeline started successfully.")  # 🔵
    APIOperation.info("Received query: 'What are your clinic hours?'")  # 🟢
    DataOperation.info("Database updated successfully.")  # 🔵
    PipelineOperation.info("Chatbot response generated.")  # 🟡

    print(f"✅ Logs are stored in: {LOG_DIR}")
