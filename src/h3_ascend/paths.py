import os
from pathlib import Path


def source_root() -> Path:
    return Path(
        os.environ.get("H3_ASCEND_SOURCE", Path(__file__).resolve().parents[2])
    ).expanduser()


def cache_root() -> Path:
    p = Path(os.environ.get("H3_ASCEND_CACHE", Path.home() / ".cache/ascend-h3")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def runtime_root(method: str) -> Path:
    p = cache_root() / "runtime" / method
    p.mkdir(parents=True, exist_ok=True)
    return p
