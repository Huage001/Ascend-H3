"""Explicit Spark defaults; no unscoped research environment switches."""
import json, os
from contextvars import ContextVar
from h3_ascend.paths import source_root
_DEFAULT = json.loads((source_root()/'configs/spark.json').read_text())
_frames = ContextVar('spark_frames', default=345)

def set_frames(frames):
    return _frames.set(frames)

def get(key, default=None):
    if key == 'MERGE_BK':
        return '256' if _frames.get() == 124 else '512'
    return os.environ.get('H3_ASCEND_SPARK_'+key, _DEFAULT.get(key, default))
