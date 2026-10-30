import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadShard:
    start: int
    count: int
    total: int = 56

    def __post_init__(self):
        if self.count <= 0 or not 0 <= self.start < self.start + self.count <= self.total:
            raise ValueError(f"invalid head shard: {self}")


@dataclass(frozen=True)
class VideoLayout:
    grid: tuple[int, int, int]
    prefix: int
    text_tokens: int = 0
    prefix_segments: tuple[int, ...] = ()
    frames: int = 124

    def __post_init__(self):
        if len(self.grid) != 3 or min(self.grid) <= 0 or self.prefix < 0:
            raise ValueError(f"invalid video layout: {self}")
        if self.frames not in (124, 345):
            raise ValueError("supported native frame counts: 124,345")
        if not 0 <= self.text_tokens <= self.prefix:
            raise ValueError("text must be within prefix")
        if self.prefix_segments and sum(self.prefix_segments) != self.prefix:
            raise ValueError("prefix segments do not cover prefix")

    @property
    def sequence_length(self):
        return self.prefix + math.prod(self.grid)

    def spark_case(self):
        f, h, w = self.grid
        return dict(
            width=w * 32,
            height=h * 32,
            frames=self.frames,
            F=f,
            grid_h=h,
            grid_w=w,
            S=h * w,
            G=self.prefix,
            text=self.text_tokens,
            seq=self.sequence_length,
        )
