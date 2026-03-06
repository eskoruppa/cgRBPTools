from __future__ import annotations
import sys

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', bar_length: int = 40) -> None:
    """
    Print a progress bar to stdout that updates in place.
    
    Parameters
    ----------
    current : int
        Current iteration (0-indexed)
    total : int
        Total iterations
    prefix : str
        Text to display before the progress bar
    suffix : str
        Text to display after the progress bar
    bar_length : int
        Length of the progress bar in characters
    """
    percent = 100 * (current / float(total))
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()