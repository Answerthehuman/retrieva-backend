import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
for path in (backend_dir, backend_dir.parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from backend.api.main import app
except ImportError:  # pragma: no cover - fallback for direct execution
    from api.main import app
