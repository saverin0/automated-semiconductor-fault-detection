import os
from pathlib import Path

def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Ensure target_path is within base_dir and does not contain suspicious elements.
    Returns True if target_path is a subpath of base_dir.
    """
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        # Add trailing slash to avoid /foo matching /foobar
        return str(target_path).startswith(str(base_dir) + os.sep) or str(target_path) == str(base_dir)
    except Exception:
        return False

def validate_env_path(var_value: str, base_dir: Path) -> Path:
    """
    Validate and return a safe Path object for environment paths.
    Raises ValueError if the path is unsafe or outside base_dir.
    """
    path = Path(var_value)
    # Prevent '..' in any part of the path
    if any(part == '..' for part in path.parts):
        raise ValueError(f"Unsafe path traversal detected: {path}")
    # Make path absolute relative to base_dir if not already
    abs_path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
    # Ensure the resolved path is within base_dir
    if not is_safe_path(base_dir, abs_path):
        raise ValueError(f"Path {abs_path} is outside of allowed base directory {base_dir}")
    return abs_path

def ensure_path_exists(path: Path, is_file: bool = False):
    """
    Ensure a directory or file exists. Create if missing.
    If is_file is True, creates parent directories and an empty file if needed.
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)