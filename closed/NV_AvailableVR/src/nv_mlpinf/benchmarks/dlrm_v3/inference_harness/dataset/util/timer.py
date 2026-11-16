"""
Timing decorator utility for performance profiling.

Provides a decorator to measure and log function execution times
with human-readable formatting.
"""

import time
from functools import wraps


def timer(func):
    """
    Decorator to measure and print function execution time.

    Args:
        func: Function to time.

    Returns:
        Wrapped function that prints execution time when called.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Format time nicely
        if elapsed_time < 1:
            print(f"⏱️  {func.__name__} completed in {elapsed_time * 1000:.2f} ms")
        elif elapsed_time < 60:
            print(f"⏱️  {func.__name__} completed in {elapsed_time:.2f} seconds")
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"⏱️  {func.__name__} completed in {minutes}m {seconds:.2f}s")

        return result
    return wrapper
