from __future__ import annotations

from pathlib import Path

def backup_filename(filepath: str | Path) -> Path:
    """Generate a backup filename by appending _#N before the extension.
    
    Args:
        filepath: Original file path (string or Path object)
        
    Returns:
        Path object with _#N appended to basename (N increments until non-existing file found)
        
    Examples:
        backup_filename('data.txt') -> Path('data_#1.txt')
        backup_filename(Path('data.txt')) -> Path('data_#1.txt')
        If data_#1.txt exists -> Path('data_#2.txt'), etc.
    """
    filepath = Path(filepath)
    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    
    counter = 1
    while True:
        backup_path = parent / f"{stem}_#{counter}{suffix}"
        if not backup_path.exists():
            return backup_path
        counter += 1