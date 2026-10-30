
import importlib as _importlib
import torch

def sparse(q, k, v, mask, sizes, qblock=128, stack=1):
    sizes64 = torch.stack((sizes.clamp(max=64), (sizes - 64).clamp(min=0)), dim=1).flatten().int()
    m = mask.repeat_interleave(2, dim=-1)
    if qblock == 64:
        m = m.repeat_interleave(2, dim=-2)
    m = m.unsqueeze(0).to(torch.int8).contiguous()
    qb, kb, vb = [x.permute(1, 0, 2).unsqueeze(0).contiguous() for x in (q, k, v)]
    result = torch.ops.sol_h3_bsa_ext.forward(qb, kb, vb, m, sizes64, [qblock, 64], None, None, 'BNSD', q.shape[1], q.shape[-1] ** (-0.5), 0, True, 128, stack)[0]
    return result.squeeze(0).transpose(0, 1).contiguous()
