import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
for path in (backend_dir, backend_dir.parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Re-exported so `uvicorn main:app` works from the repo root as well as from
# inside backend/. Not used in this module — hence the noqa.
try:
    from backend.api.main import app  # noqa: F401
except ImportError:  # pragma: no cover - fallback for direct execution
    pass
