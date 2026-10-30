
"""Patched copy of the Sol + Veda fused kernel; see attention.py."""
import importlib as _importlib
from .attention import attend
__all__ = ['attend']
