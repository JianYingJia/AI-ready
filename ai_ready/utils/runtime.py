"""Set project-local caches before importing model or Data-Juicer packages."""

import os
import sys
from pathlib import Path


def configure_runtime(cache_dir: Path) -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11 in conda environment aiReady")
    cache_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(cache_dir / "datasets"),
        "TORCH_HOME": str(cache_dir / "torch"),
        "MODELSCOPE_CACHE": str(cache_dir / "modelscope"),
        "UV_CACHE_DIR": str(cache_dir / "uv"),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "4",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    for key, value in defaults.items():
        if key.endswith("CACHE") or key.endswith("CACHE_DIR") or key.endswith("HOME"):
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    # Data-Juicer's optional installer invokes uv without --python. Explicitly
    # target this interpreter so it cannot install into base or another env.
    os.environ["UV_PYTHON"] = sys.executable
    os.environ["AI_READY_CACHE_DIR"] = str(cache_dir)
