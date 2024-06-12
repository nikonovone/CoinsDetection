import logging
from functools import wraps


def handle_exceptions(action_description: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Failed to {action_description}: {e}"
                logging.error(
                    error_msg,
                    exc_info=True,
                )  # Log the exception with traceback
                raise RuntimeError(error_msg) from e

        return wrapper

    return decorator


@handle_exceptions("Initialize logger")
def init_logger(
    logger_name: str = "__main__",
    log_level: str = "INFO",
    all_log_level: str = "CRITICAL",
):
    """
    Initialize a logger with specified log levels and configuration.

    Args:
        logger_name (str): The name of logger. Default is "__main__".
        log_level (str): The log level for the logger. Default is "INFO".
        all_log_level (str): The log level for all loggers. Default is "CRITICAL".

    Returns:
        logger (logging.Logger): The initialized logger instance.
    """

    logger = logging.getLogger(logger_name)

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    for logger_name in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_levels.get(all_log_level, logging.INFO))

    logger.setLevel(log_levels.get(log_level, logging.INFO))

    # Add a StreamHandler to the logger to send log messages to the console
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
