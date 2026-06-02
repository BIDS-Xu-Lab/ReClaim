#!/usr/bin/env python3

# Set timezone to Eastern Standard Time
import os
import time
os.environ['TZ'] = 'America/New_York'
time.tzset()  # Apply the timezone change

import sys
import logging
import inspect
from datetime import datetime


INDENT = "    "


class StreamToLogger:
    """Redirect print() output to logger"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level 
        self.linebuf = ''
    
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def setup_logging(script_path=None, capture_print=True, level=logging.INFO):
    """
    Set up logging to file and console with Eastern Time timestamps
    
    Args:
        script_path (str, optional): Path to the script. If None, __file__ will be used
                                    from the calling script.
        capture_print (bool): Whether to redirect print() to logger. Default True.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
                    Default is logging.INFO.
    
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get the script name without extension
    if script_path is None:
        # Get the frame of the caller
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_path = module.__file__
    
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Generate default log filename with script name and timestamp
    # Since we set TZ=America/New_York at the top of the file,
    # datetime.now() will automatically use Eastern Time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{script_name}_{timestamp}.log"
    
    # Configure logging to use Eastern Time
    # force=True ensures this works even if logging was already configured by imports
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True,
    )

    logger = logging.getLogger()

    # redirect print() to logger if requested
    if capture_print:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger.info("Logger initialized: %s (print capture enabled)", log_file)
    else:
        logger.info("Logger initialized: %s", log_file)
    
    return log_file


def format_time_delta(seconds):
    """
    Format time delta in a readable format
    
    Args:
        seconds (float): Number of seconds
    
    Returns:
        str: Formatted time string (e.g. "1h 30m 45s" or "45.23s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.2f}s"


def time_execution(func):
    """
    Decorator to measure and log the execution time of a function
    
    Args:
        func: The function to be timed
    
    Returns:
        The wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        formatted_time = format_time_delta(execution_time)
        logging.info("Function '%s' executed in %s", func.__name__, formatted_time)
        return result
    return wrapper


