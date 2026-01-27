from __future__ import annotations
from pathlib import Path

def create_relative_path(filename: str | Path) -> None:
    """Create directory path if it doesn't exist.
    
    Args:
        filename: File path (str or Path) whose parent directory should be created.
    """
    path = Path(filename).parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)