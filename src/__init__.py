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

# Now you have the main project directory path
# Define the log directory relative to the project folder
LOG_DIR = MAIN_DIR / "logs"
LOG_DIR = os.path.abspath(LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)

# Create a function to set up the logger
def setup_logger(name, log_file, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a color handler for the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a colored log formatter
    formatter = colorlog.ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    # Create loggers with color output
    HighLevelErrors = setup_logger(name="HighLevelErrors", log_file=os.path.join(LOG_DIR, "HighLevelErrors.log"), level=logging.ERROR)
    ModelingOperation = setup_logger(name="ModelingOperation", log_file=os.path.join(LOG_DIR, "ModelingOperation.log"), level=logging.WARNING)
    APIOperation = setup_logger(name="APIOperation", log_file=os.path.join(LOG_DIR, "APIOperation.log"), level=logging.WARNING)
    DataOperation = setup_logger(name="DataOperation", log_file=os.path.join(LOG_DIR, "DataOperation.log"), level=logging.WARNING)
    PipelineOperation = setup_logger(name="PipelineOperation", log_file=os.path.join(LOG_DIR, "PipelineOperation.log"), level=logging.INFO)
    # Example logging calls
    HighLevelErrors.error("This is an error message.")
    ModelingOperation.info("This is an info message.")
    APIOperation.warning("This is a warning message.")
    DataOperation.warning("This is a warning message.")
    PipelineOperation.warning("This is a warning message.")
