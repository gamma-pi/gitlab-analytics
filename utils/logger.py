import logging
from logging.handlers import TimedRotatingFileHandler
import time
from datetime import datetime
import os
import sys

sys.path.append(os.getcwd())

def set_logg_handlers_to_console_and_file(log_folder):
    """
    Sets up logging handlers to write logs to both console and file.
    """
    try:
        # Ensure the log folder exists
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Configure timestamped log file
        current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'MR_approvals_automate_{current_timestamp}.log'
        log_file_path = os.path.join(log_folder, log_filename)

        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)  # Set the logger level

        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    except Exception as e:
        print("Could not init Logger")
        raise e



def log_function_data(logger):
    """
    Decorator to log the execution details of a function.

    This decorator logs:
    1. The start time of the function execution.
    2. The end time of the function execution.
    3. The duration of the function execution in seconds.

    Parameters:
        logger (logging.Logger): The logger instance used to log the function execution details.

    Returns:
        function: A wrapper function that logs the start time, end time, and duration of the wrapped function.

    Example:
        @log_function_data(logger)
        def my_function():
            # Function implementation
            pass
    Notes:
        - Ensure that the `logger` passed to the decorator is properly configured.
        - The `logger` should be an instance of `logging.Logger`.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Function {func.__name__} started at {start_timestamp}")
            result = func(*args, **kwargs)
            end_time = time.time()
            end_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            duration = end_time - start_time
            logger.info(f"Function {func.__name__} ended at {end_timestamp}")
            logger.info(f"Function {func.__name__} duration: {duration:.2f} seconds")
            return result
        return wrapper
    return decorator

