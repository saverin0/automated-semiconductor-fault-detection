import os
import re

def scan_for_hardcoded_paths(root_dir):
    pattern = re.compile(r'(["\'])(/[^"\']+|[A-Za-z0-9_\-/]+\.log|\.csv|\.json)\1')
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(subdir, file)
                with open(path, 'r') as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            print(f"{path}:{i}: {line.strip()}")

if __name__ == "__main__":
    scan_for_hardcoded_paths("./src")