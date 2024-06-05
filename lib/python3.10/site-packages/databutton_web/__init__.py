from pathlib import Path

__version__ = '0.13.2'

def get_static_file_path():
    app_path = Path(__file__).parent / "local"
    return str(app_path)
    