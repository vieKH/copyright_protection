"""Create independent experiment directories without deleting prior results."""
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

def create_run_directory(base, image_path, qr_size, block_size, gap_size, param_name=None, param_value=None):
    base = Path(base)
    image_name = Path(image_path).stem
    date_str = datetime.now().strftime("%Y-%m-%d")

    experiment_name = (
        f"{date_str}_"
        f"{image_name}_"
        f"{qr_size}_"
        f"{block_size}_"
        f"{gap_size}"
    )

    output = base / experiment_name

    if param_name is not None and param_value is not None:
        value = f"{float(param_value):.6f}".rstrip("0").rstrip(".")
        output = output / f"{param_name}_{value}"

    output.mkdir(parents=True, exist_ok=True)
    return str(output)