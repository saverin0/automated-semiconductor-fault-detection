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
    return (base_dir / path) if not path.is_absolute() else path

def ensure_path_exists(path: Path, is_file: bool = False):
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)