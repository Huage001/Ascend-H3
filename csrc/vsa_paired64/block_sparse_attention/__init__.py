from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers the PrivateUse1/NPU backend


_LIBRARY = Path(__file__).resolve().parent / "lib" / "libblock_sparse_attention.so"
if not _LIBRARY.is_file():
    raise ImportError(
        f"native BlockSparseAttention library was not found at {_LIBRARY}; "
        "build/install the wheel before importing the package"
    )
torch.ops.load_library(str(_LIBRARY))

from .functional import (  # noqa: E402
    block_sparse_attention,
    block_sparse_attention_backward,
    npu_block_sparse_attention,
)

__all__ = [
    "block_sparse_attention",
    "block_sparse_attention_backward",
    "npu_block_sparse_attention",
]

__version__ = "0.1.0"
