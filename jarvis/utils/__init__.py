import functools
import logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARN
from typing import Callable


def log(
    func: Callable,
    level: INFO | DEBUG | WARN | CRITICAL | ERROR = INFO,
    logger: logging.Logger = logging.getLogger(),
):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug(f"Calling {func.__name__}({signature})")

        try:
            value = func(*args, **kwargs)
            error = None
        except Exception as e:
            error = e
            value = None

        if error:
            logger.log(level, f"Exception [{func.__name__}] ({signature}) -> ({error})")
            raise error
        else:
            logger.log(level, f"Executed [{func.__name__}] ({signature}) -> ({value})")
            return value

    return wrapper
