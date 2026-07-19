from functools import wraps
from typing import Callable
import logging

from autonerves.conf import instance

logger = logging.getLogger(__name__)


def should_output(name: str) -> bool:
    """
    Determine whether a file with a given name (excluding extension) should be output.

    This is configured in config/output.yaml. If the file is not present in the config, the default value is used.

    Parameters
    ----------
    name
        The name of the file to be output, excluding extension.

    Returns
    -------
    Whether the file should be output.
    """
    output_config = instance["output"]
    try:
        return output_config[name]
    except KeyError:
        return output_config["default"]


def conditional_output(func: Callable):
    """
    Decorator for functions that output files. If the file should not be output, the function is not called.

    Parameters
    ----------
    func
        A method where the first argument is the name of the file to be output.

    Returns
    -------
    The decorated function.
    """

    @wraps(func)
    def wrapper(self, name: str, *args, **kwargs):
        """
        Conditionally call the decorated function if the file should be output according
        to the config.

        Parameters
        ----------
        self
        name
            The name of the file to be output, excluding extension.
        args
        kwargs
        """
        if should_output(name):
            return func(self, name, *args, **kwargs)
        logger.info(f"Skipping output of {name}")

    return wrapper
