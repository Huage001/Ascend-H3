"""Full-width FP32 DeltaNet state with NPU-friendly factorizations and scans.

Newton iteration is a numerical inverse approximation, not bitwise Cholesky.
The default keeps all 128 key coordinates and targets a 1e-8 ideal bound.
"""

import os


def iterations(upper, tolerance=1e-8):
    r = (upper / (2 + upper)) ** 2
    k = 1
    while r > tolerance and k < 24:
        r = (r / (2 - r)) ** 2
        k += 1
    return k


def delta_factors(a, b, alpha):
    import torch

    if a.dtype != torch.float32:
        a = a.float()
    if a.shape[-1] != 128:
        raise ValueError("VDN default retains the full 128-dimensional state")
    eye = torch.eye(128, device=a.device, dtype=torch.float32)
    trace = a.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0)
    upper = (
        torch.minimum(
            trace,
            torch.minimum(a.square().sum((-1, -2)).sqrt(), a.abs().sum(-1).amax(-1)),
        )
        * 1.0001
    )
    tol = float(os.environ.get("H3_ASCEND_VDN_INVERSE_TOL", "1e-8"))
    cap = os.environ.get("H3_ASCEND_VDN_INVERSE_STEPS", "auto")
    text = a.shape[0] == 1
    n = (
        iterations(float(upper.max().item()), 1e-8 if text else tol)
        if cap == "auto" or text
        else int(cap)
    )
    m = a + eye
    x = eye / ((2 + upper) * 0.5)[..., None, None]
    r = (upper / (2 + upper)).square()
    for i in range(n):
        if i:
            x = x * (2 / (2 - r))[..., None, None]
            r = (r / (2 - r)).square()
        x = x + x @ (eye - m @ x)
        x = 0.5 * (x + x.transpose(-1, -2))
    return alpha[..., None] * x, b.float() @ x


def plain_scans(t, b, start):
    import torch

    p = torch.empty_like(b)
    s = torch.empty_like(b)
    state = start
    for f in range(len(t)):
        torch.baddbmm(b[f], state, t[f], out=p[f])
        state = p[f]
    state = start
    for f in range(len(t) - 1, -1, -1):
        torch.baddbmm(b[f], state, t[f], out=s[f])
        state = s[f]
    return p, s


class _Scan:
    def __init__(self, t, b, start):
        import torch

        self.inputs = [v.contiguous().clone() for v in (t, b, start)]
        stream = torch.npu.Stream(device=t.device)
        stream.wait_stream(torch.npu.current_stream(t.device))
        with torch.npu.stream(stream):
            for _ in range(2):
                expected = plain_scans(*self.inputs)
        torch.npu.current_stream(t.device).wait_stream(stream)
        torch.npu.synchronize(t.device)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=stream):
            self.outputs = plain_scans(*self.inputs)
        self.graph.replay()
        torch.npu.synchronize(t.device)
        if not all(torch.equal(x, y) for x, y in zip(self.outputs, expected)):
            raise RuntimeError("VDN scan capture differs from eager")

    def __call__(self, t, b, start):
        for dst, src in zip(self.inputs, (t, b, start)):
            dst.copy_(src)
        self.graph.replay()
        return self.outputs


_SCANS = {}


def scans(t, b, start):
    import torch

    key = (
        tuple(t.shape),
        str(t.device),
        int(torch.npu.current_stream(t.device).npu_stream),
    )
    if key not in _SCANS:
        _SCANS[key] = _Scan(t, b, start)
    return _SCANS[key](t, b, start)


def packed_window_batches(cq, ck, element_bytes, max_gather_bytes=1024**3):
    """Partition complete packed sequences while bounding each KV gather."""
    if len(cq) != len(ck):
        raise ValueError("Packed Q and KV window counts differ")
    first = 0
    while first < len(cq):
        qlo = cq[first - 1] if first else 0
        klo = ck[first - 1] if first else 0
        last = first + 1
        while last < len(cq) and (ck[last] - klo) * element_bytes <= max_gather_bytes:
            last += 1
        yield first, last, qlo, cq[last - 1], klo, ck[last - 1]
        first = last


def batched_window(query, key, value, window):
    """Bounded packed-window groups plus global rows, retaining every key."""
    import torch
    import torch_npu
    from vllm_omni.diffusion.attention.backends.vdnh3_attn import window_plan

    plan = window_plan(window, query.device)
    gkey = (window, str(query.device))
    if gkey not in _WINDOW_PLANS:
        dense = []
        qw = []
        kw = []
        cq = []
        ck = []
        nq = nk = 0
        for lo, hi, keys in plan:
            rows = torch.arange(lo, hi, device=query.device)
            if keys is None:
                dense.append(rows)
            else:
                qw.append(rows)
                kw.append(keys)
                nq += hi - lo
                nk += keys.numel()
                cq.append(nq)
                ck.append(nk)
        empty = torch.empty(0, device=query.device, dtype=torch.long)
        _WINDOW_PLANS[gkey] = (
            torch.cat(dense) if dense else empty,
            torch.cat(qw) if qw else empty,
            torch.cat(kw) if kw else empty,
            cq,
            ck,
        )
    dq, wq, wk, cq, ck = _WINDOW_PLANS[gkey]
    q, k, v = [z[0, : window.used].contiguous() for z in (query, key, value)]
    h = q.shape[1]
    scale = q.shape[-1] ** -0.5
    out = torch.zeros_like(query)
    if dq.numel():
        y = torch_npu.npu_fused_infer_attention_score(
            q.index_select(0, dq)[None],
            k[None],
            v[None],
            num_heads=h,
            num_key_value_heads=h,
            input_layout="BSND",
            scale=scale,
        )[0][0]
        out[0].index_copy_(0, dq, y)
    if wq.numel():
        # Packed windows are independent softmax sequences. Limit each KV
        # gather to 1 GiB, retaining every token and each window's full keys.
        # This changes batching only, never splits a softmax reduction.
        element_bytes = k.element_size() * k.shape[1] * k.shape[2]
        for first, last, qlo, qhi, klo, khi in packed_window_batches(cq, ck, element_bytes):
            selected_q = wq[qlo:qhi]
            qpart = q.index_select(0, selected_q)
            kpart = k.index_select(0, wk[klo:khi])
            vpart = v.index_select(0, wk[klo:khi])
            y = torch_npu.npu_fusion_attention(
                qpart,
                kpart,
                vpart,
                h,
                "TND",
                scale=scale,
                actual_seq_qlen=[n - qlo for n in cq[first:last]],
                actual_seq_kvlen=[n - klo for n in ck[first:last]],
            )[0]
            out[0].index_copy_(0, selected_q, y)
            del y, qpart, kpart, vpart
    return out


_WINDOW_PLANS = {}

_FEATURES = {}
_ACTIVATIONS = {}


def activation(tokens, normalize):
    import torch
    import torch.nn.functional as F

    key = (tuple(tokens.shape), normalize, str(tokens.device))
    if key not in _ACTIVATIONS:

        def run(x):
            z = F.silu(x.float())
            return (F.normalize(z, dim=-1, eps=1e-6) if normalize else z).to(x.dtype)

        from torch_npu.dynamo import torchair

        _ACTIVATIONS[key] = torch.compile(
            run, backend=torchair.get_npu_backend(), dynamic=False, fullgraph=True
        )
    return _ACTIVATIONS[key](tokens)


def fused_features(branch, proj, tokens, grid):
    import torch
    import torch.nn.functional as F

    conv = grid is not None and branch.short_conv is not None and proj in branch.short_conv.targets
    if not conv:
        return activation(tokens, proj != "v")
    f, h, w = grid
    channels = tokens.shape[1] * tokens.shape[2]
    ws = getattr(branch.short_conv, f"{proj}_sp").weight
    wt = getattr(branch.short_conv, f"{proj}_tm").weight
    engine = os.environ.get("H3_ASCEND_VDN_TEMPORAL", "auto")
    if engine not in ("auto", "native", "ge"):
        raise ValueError("H3_ASCEND_VDN_TEMPORAL must be auto, native or ge")
    if engine == "native" or (engine == "auto" and grid != (37, 12, 21)):
        spatial = (
            F.conv2d(
                tokens.reshape(f, h, w, channels).permute(0, 3, 1, 2),
                ws,
                padding=2,
                groups=channels,
            )
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(f, h * w, channels)
        )
        # Spatial positions become the Conv2D batch; time is its height.
        temporal = F.conv2d(
            spatial.permute(1, 0, 2).contiguous().view(h * w, f, 1, channels).permute(0, 3, 1, 2),
            wt.view(channels, 1, 5, 1),
            padding=(2, 0),
            groups=channels,
        )
        z = temporal.permute(0, 2, 3, 1).squeeze(2).permute(1, 0, 2).contiguous().view_as(tokens)
        return activation(z, proj != "v")
    key = (grid, channels, proj != "v")
    if key not in _FEATURES:

        def run(x, sp, tm):
            vol = x.reshape(f, h, w, channels).permute(0, 3, 1, 2)
            vol = F.conv2d(vol, sp, padding=2, groups=channels)
            pad = F.pad(vol.permute(0, 2, 3, 1).float(), (0, 0, 0, 0, 0, 0, 2, 2))
            taps = tm.float().squeeze(1)
            z = sum(pad[i : i + f] * taps[:, i] for i in range(5)).reshape(x.shape)
            z = F.silu(z)
            return (F.normalize(z, dim=-1, eps=1e-6) if proj != "v" else z).to(x.dtype)

        if os.getenv("H3_ASCEND_VDN_FEATURE_FUSION", "1") == "1":
            from torch_npu.dynamo import torchair

            run = torch.compile(
                run, backend=torchair.get_npu_backend(), dynamic=False, fullgraph=True
            )
        _FEATURES[key] = run
    return _FEATURES[key](tokens, ws, wt)


def install():
    """Patch the upstream learned branch; checkpoint/TP/INT8 remain upstream."""
    import torch
    from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNAttentionImpl

    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    from vllm_omni.diffusion.models.minimax_h3 import vdnh3 as model

    model._delta_factors = delta_factors
    model.VDNLinearBranch._features = fused_features

    @torch.compiler.disable
    def state(self, k, v, beta, frame_mean, text, bounds):
        text_state = self._text_state(*text) if text is not None else None
        a, b = model._frame_statistics(k, v, beta)
        alpha = self.alpha(frame_mean)
        t, inj = delta_factors(a, b, alpha)
        start = torch.zeros_like(inj[0]) if text_state is None else text_state
        prefix, suffix = scans(t, inj, start)
        return model._outside_window_state(
            prefix,
            suffix,
            alpha,
            bounds,
            text_state=text_state,
            bridge=self.config.bridge,
        ).to(k.dtype)

    model.VDNLinearBranch._state = state

    @torch.compiler.disable
    def npu_forward(self, q, k, v, attn_metadata=None):
        win = attn_metadata.extra.get("vdn_window") if attn_metadata else None
        if win is None:
            from h3_ascend.integrations.vllm_omni import _dense

            return _dense(q, k, v, attn_metadata)
        return batched_window(q, k, v, win)

    VDNAttentionImpl.forward_npu = npu_forward
