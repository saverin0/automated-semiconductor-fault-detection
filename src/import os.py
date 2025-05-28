import os
import re

def scan_for_hardcoded_paths(root_dir):
    # Match absolute or relative paths ending with .log, .csv, .json, or starting with /
    pattern = re.compile(r'(["\'])(/[^"\']+|\.{0,2}/[^"\']+|[A-Za-z0-9_\-/]+\.log|[A-Za-z0-9_\-/]+\.csv|[A-Za-z0-9_\-/]+\.json)\1')
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(subdir, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        in_docstring = False
                        for i, line in enumerate(f, 1):
                            # Skip comments and docstrings
                            stripped = line.strip()
                            if stripped.startswith('"""') or stripped.startswith("'''"):
                                in_docstring = not in_docstring
                                continue
                            if in_docstring or stripped.startswith('#'):
                                continue
                            match = pattern.search(line)
                            if match:
                                print(f"{path}:{i}: {line.strip()}  [MATCH: {match.group(0)}]")
                except Exception as e:
                    print(f"Could not read {path}: {e}")

if __name__ == "__main__":
    scan_for_hardcoded_paths("./src")