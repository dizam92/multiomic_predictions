import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import List


FORMATTER = logging.Formatter("[%(levelname)s] %(message)s")

handlers: List[logging.Handler] = []
level = None
start_time = None


def sec_since_init():
    return time.time() - start_time  # type: ignore


def initialize(file_name=None, debug=False, std=False):
    """Initialize the logging module.
    It can be called before any logger is created to change the default arguments.
    """
    global start_time
    start_time = time.time()
    if file_name is not None:
        _initialize_handler_file(file_name)
        if std:
            _initialize_handler_std()
    else:
        _initialize_handler_std()

    _initialize_level(debug)


def _initialize_handler_file(file_name):
    # Create a handler_file that rotate files each 512mb
    handler_file = RotatingFileHandler(
        file_name, mode="a", maxBytes=536_870_912, backupCount=4, encoding=None
    )
    handler_file.setFormatter(FORMATTER)
    handlers.append(handler_file)


def _initialize_handler_std():
    handler_std = logging.StreamHandler(stream=sys.stdout)
    handler_std.setFormatter(FORMATTER)
    handlers.append(handler_std)


def _initialize_level(debug):
    global level
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO


def create_logger(name: str) -> logging.Logger:
    """Create a logger with default configuration and FORMATTER."""
    initialized = level is not None and len(handlers) > 0
    if not initialized:
        initialize()

    logger = logging.getLogger(name)
    logger.setLevel(level)  # type: ignore

    for handler in handlers:
        logger.addHandler(handler)

    return logger
