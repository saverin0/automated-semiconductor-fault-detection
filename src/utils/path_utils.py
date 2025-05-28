import os
from pathlib import Path

def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()
        return str(target_path).startswith(str(base_dir))
    except Exception:
        return False

def validate_env_path(var_value: str, base_dir: Path) -> Path:
    path = Path(var_value)
    if path.is_absolute() and not str(path).startswith(str(base_dir)):
        raise ValueError(f"Unsafe absolute path detected: {path}")
    if any(part == '..' for part in path.parts):
        raise ValueError(f"Unsafe path traversal detected: {path}")
    abs_path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if not is_safe_path(base_dir, abs_path):
        raise ValueError(f"Path {abs_path} is outside of allowed base directory {base_dir}")
    return abs_path

def ensure_path_exists(path: Path, is_file: bool = False):
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)