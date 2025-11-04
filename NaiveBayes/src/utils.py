import os


def ensure_directories():
    """Create project-level folders (data, results, notebooks) under cwd.

    Previously relative paths used `..` which created folders in the parent
    directory and caused permission/path issues. Use the current working
    directory as the base so `data/` and `results/` are created next to
    `main.py` when the script is run from the project root.
    """
    base = os.getcwd()
    for folder in ["data", "results", "notebooks"]:
        os.makedirs(os.path.join(base, folder), exist_ok=True)
