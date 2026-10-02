"""Create independent experiment directories without deleting prior results."""
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp


def create_run_directory(base, gap):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    prefix = f"gap_{gap}_{datetime.now():%Y%m%d_%H%M%S}_"
    return mkdtemp(prefix=prefix, dir=str(base))
