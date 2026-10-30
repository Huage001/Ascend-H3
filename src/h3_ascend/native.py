"""Lazy loading of user-built CANN operators."""

from h3_ascend.paths import cache_root

_LOADED = set()
LIBRARIES = {
    "vsa_paired64": "libblock_sparse_attention.so",
    "veda_cached128": "libmiowtion128cache_attention.so",
    "veda_schedule128": "libmiowtion128schedule_attention.so",
    "veda_schedule2": "libmiowtion128schedule2_attention.so",
    "veda_pipe3": "libmiowtion128pipe3_attention.so",
}


def library_path(name):
    return cache_root() / "lib" / name / LIBRARIES[name]


def load(name):
    path = library_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run: h3-ascend build --method all")
    if str(path) not in _LOADED:
        import torch

        torch.ops.load_library(str(path))
        _LOADED.add(str(path))
