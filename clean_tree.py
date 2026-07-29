import os
import sys

SKIP_DIRS = {
    'node_modules', '__pycache__', '.git', 'venv', '.venv', 'env',
    'dist', 'build', '.pytest_cache', '.mypy_cache', '.idea', '.vscode',
    '.tox', '*.egg-info', 'htmlcov', '.coverage'
}
SKIP_EXTS = {'.pyc', '.pyo', '.log', '.tmp', '.coverage'}

def show_tree(path='.', prefix='', max_depth=4, current_depth=0):
    if current_depth >= max_depth:
        return

    try:
        entries = sorted(
            os.scandir(path),
            key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        return

    # Filter out junk
    entries = [e for e in entries
               if e.name not in SKIP_DIRS
               and not any(e.name.endswith(ext) for ext in SKIP_EXTS)
               and not e.name.startswith('.')]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = '└── ' if is_last else '├── '
        icon = '📁' if entry.is_dir() else '📄'
        print(f"{prefix}{connector}{icon} {entry.name}")

        if entry.is_dir():
            extension = '    ' if is_last else '│   '
            show_tree(entry.path, prefix + extension, max_depth, current_depth + 1)

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    print(f"\n📂 {os.path.abspath(root)}")
    print("=" * 50)
    show_tree(root)
