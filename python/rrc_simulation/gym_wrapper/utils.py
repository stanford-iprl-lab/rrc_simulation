import numpy as np
import time
import datetime

import inspect
from typing import Any, Dict, Optional, Type


def scale(x, space):
    """
    Scale some input to be between the range [-1;1] from the range
    of the space it belongs to
    """
    return 2.0 * (x - space.low) / (space.high - space.low) - 1.0


def unscale(y, space):
    """
    Unscale some input from [-1;1] to the range of another space
    """
    return space.low + (y + 1.0) / 2.0 * (space.high - space.low)


def compute_distance(a, b):
    """
    Returns the Euclidean distance between two
    vectors (lists/arrays)
    """
    return np.linalg.norm(np.subtract(a, b))


def sleep_until(until, accuracy=0.01):
    """
    Sleep until the given time.

    Args:
        until (datetime.datetime): Time until the function should sleep.
        accuracy (float): Accuracy with which it will check if the "until" time
            is reached.

    """
    while until > datetime.datetime.now():
        time.sleep(accuracy)


def configurable(pickleable: bool = False):
    """Class decorator to allow injection of constructor arguments.

    Example usage:
    >>> @configurable()
    ... class A:
    ...     def __init__(self, b=None, c=2, d='Wow'):
    ...         ...

    >>> set_env_params(A, {'b': 10, 'c': 20})
    >>> a = A()      # b=10, c=20, d='Wow'
    >>> a = A(b=30)  # b=30, c=20, d='Wow'

    Args:
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the constructor arguments.
    """
    # pylint: disable=protected-access,invalid-name

    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__

        def __init__(self, *args, **kwargs):

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'
            def __getstate__(self):
                return {
                   PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }
            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                inst = type(self)(*saved_args, **saved_kwargs)
                self.__dict__.update(inst.__dict__)

            cls.__setstate__ = __setstate__

        return cls

    # pylint: enable=protected-access,invalid-name
    return cls_decorator
