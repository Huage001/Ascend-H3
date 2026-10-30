
"""Prepared dynamic NPU reblock graph, shared across layers of one request shape.
Only workspace and launch topology are cached. Every replay rebuilds token
partitions from the current transformed Q/K feature values.
"""
import importlib as _importlib
from . import settings as _settings
import torch

class PreparedTree:

    def __init__(self, h, n, device):
        self.input = torch.empty((2 * h, n, 128), device=device, dtype=torch.bfloat16)
        self.graph = None
        self.output = None

    def __call__(self):
        if self.graph is None:
            from .backend import tree
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(2):
                    ref = tree(self.input)
                ref = ref.clone()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, stream=stream):
                self.output = tree(self.input)
            self.graph.replay()
            torch.npu.synchronize()
            assert torch.equal(self.output, ref), 'NPU reblock graph capture differs from eager'
        else:
            self.graph.replay()
        return self.output
