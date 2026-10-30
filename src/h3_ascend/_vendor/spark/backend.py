
"""Default Spark-H3 math port. Content-dependent routes recomputed on every call."""
import importlib as _importlib
from . import settings as _settings
import torch
import time
import os
from . import kernels
from h3_ascend._vendor.spark_official.mahalanobis_kmeans import hilbert_midpoint_sample_indices
from h3_ascend._vendor.spark_official.reblock_hierarchy import power_of_two_child_budgets

def unit(x):
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)

def directions(centers, weights, caps):
    x = centers.float()
    b, l, d = x.shape
    u = unit(x)
    dist = 1 - torch.bmm(u, u.transpose(1, 2))
    upper = torch.ones((l, l), device=x.device, dtype=torch.bool).triu(1)
    active = weights.long()[:, None]
    ranges = [(0, len(caps))]
    dirs = []
    while ranges:
        ns = len(ranges)
        valid = active > 0
        pair = dist[:, None].masked_fill(~(valid[:, :, :, None] & valid[:, :, None, :] & upper), -float('inf')).flatten(2).argmax(2)
        only = valid.int().argmax(2)
        one = valid.sum(2) < 2
        first = torch.where(one, only, pair // l)
        second = torch.where(one, only, pair % l)
        expanded = x[:, None].expand(-1, ns, -1, -1)
        left = expanded.gather(2, first[..., None, None].expand(-1, -1, 1, 128)).squeeze(2)
        right = expanded.gather(2, second[..., None, None].expand(-1, -1, 1, 128)).squeeze(2)
        lc = torch.tensor([sum(caps[s:(s + e) // 2]) for s, e in ranges], device=x.device)
        rc = torch.tensor([sum(caps[(s + e) // 2:e]) for s, e in ranges], device=x.device)
        for _ in range(2):
            scores = torch.bmm(u, (unit(right) - unit(left)).transpose(1, 2)).transpose(1, 2)
            order = scores.argsort(dim=2, stable=True)
            ordered = active.gather(2, order)
            before = ordered.cumsum(2) - ordered
            take = torch.minimum((lc[None, :, None] - before).clamp_min(0), ordered)
            lw = torch.zeros_like(active).scatter(2, order, take)
            rw = active - lw
            left = torch.bmm(lw.float(), x) / lc[None, :, None]
            right = torch.bmm(rw.float(), x) / rc[None, :, None]
        dirs.append(unit(right) - unit(left))
        active = torch.stack((lw, rw), dim=2).reshape(b, ns * 2, l)
        ranges = [(a, b) for s, e in ranges for a, b in ((s, (s + e) // 2), ((s + e) // 2, e)) if b - a > 1]
    return torch.cat(dirs, 1).contiguous()

def split(ids, scores, caps, positions=None):
    if _settings.get('POSITION_ONLY') == '1':
        return _importlib.import_module('h3_ascend._vendor.spark.partition_positions').split(ids, scores, caps)
    out = []
    front = [(ids, positions if positions is not None else torch.arange(ids.shape[1], device=ids.device, dtype=ids.dtype)[None].expand(ids.shape[0], -1), 0, 0, len(caps))]
    while front:
        cur, pos, node, s, e = front.pop(0)
        mid = (s + e) // 2
        take = sum(caps[s:mid])
        v = _importlib.import_module('h3_ascend._vendor.spark.meta32_native').gather(scores[:, :, node], pos) if ids.dtype == torch.int32 else scores[:, :, node].gather(1, pos)
        if _settings.get('PARTITION', 'sort') in ('prefix', 'cann'):
            oi, op = (_importlib.import_module('h3_ascend._vendor.spark.meta32_native').partition(cur, pos, v, take) if ids.dtype == torch.int32 else _importlib.import_module('h3_ascend._vendor.spark.preproc_native').partition(cur, pos, v, take)) if _settings.get('PARTITION') == 'cann' else _importlib.import_module('h3_ascend._vendor.spark.partition_prefix').partition(cur, pos, v, take)
            parts = ((oi[:, :take], op[:, :take], s, mid, 2 * node + 1), (oi[:, take:], op[:, take:], mid, e, 2 * node + 2))
        else:
            order = v.argsort(dim=1, stable=True)
            left_order = order[:, :take].float().sort(1).values.long()
            right_order = order[:, take:].float().sort(1).values.long()
            parts = tuple(((cur.gather(1, o), pos.gather(1, o), a, b, ch) for o, a, b, ch in ((left_order, s, mid, 2 * node + 1), (right_order, mid, e, 2 * node + 2))))
        for c, p, a, b, ch in parts:
            if b - a == 1:
                out.append((a, c))
            else:
                front.append((c, p, ch, a, b))
    return torch.cat([v for _, v in sorted(out)], 1)

def tree(x, precision='fp16'):
    b, n, d = x.shape
    a = n // 64 * 64
    rem = n - a
    score_source = None
    packed_norm = None
    if rem and _settings.get('ROOT_NORM_PACK') == '1' and (_settings.get('SCORE_PREUNIT') == '1'):
        score_source, packed_norm = _importlib.import_module('h3_ascend._vendor.spark.scores_preunit').pack_norm(x.reshape(-1, 128))
        packed_norm = packed_norm.view(b, n)
    meta32 = _settings.get('TREE_META32') == '1'
    if meta32:
        assert b * n < 2 ** 31, 'tree metadata exceeds signed32-bit range'
    ids = torch.arange(n, device=x.device, dtype=torch.int32 if meta32 else torch.int64).expand(b, -1)
    if rem:
        norm = packed_norm if packed_norm is not None else x.float().square().sum(-1) if _settings.get('NORM', 'torch') == 'torch' else _importlib.import_module('h3_ascend._vendor.spark.feature_kernels').norm(x)
        if _settings.get('ROOT_PARTITION') == 'cann':
            root, excluded = _importlib.import_module('h3_ascend._vendor.spark.meta32_native').root_partition(norm, a) if meta32 else _importlib.import_module('h3_ascend._vendor.spark.preproc_native').root_partition(norm, a)
        else:
            threshold = norm.topk(rem, dim=1, sorted=False).values.amin(1, keepdim=True)
            greater = norm > threshold
            equal = norm == threshold
            need = rem - greater.sum(1, keepdim=True)
            mask = (greater | equal & (equal.int().cumsum(1) <= need)).int()
            if _settings.get('ROOT_PARTITION', 'sort') == 'prefix':
                rank = mask.cumsum(1, dtype=torch.int32)
                dest = torch.where(mask.bool(), a + rank - 1, ids - rank)
                root = torch.empty_like(ids).scatter_(1, dest.long(), ids)
            else:
                root = mask.float().argsort(dim=1, stable=True)
            excluded = root[:, a:]
            root = root[:, :a]
    else:
        root = ids
        excluded = ids[:, :0]
    output = torch.empty_like(ids)
    source = x.reshape(-1, 128)
    heads = torch.arange(b, device=x.device, dtype=ids.dtype)
    if score_source is None:
        score_source = _importlib.import_module('h3_ascend._vendor.spark.scores_preunit').pack(source) if _settings.get('SCORE_PREUNIT') == '1' else None
    frontier = [(root, heads, torch.zeros_like(heads), a // 64)]
    depth = 0
    while frontier:
        groups = {}
        for item in frontier:
            groups.setdefault(item[-1], []).append(item)
        next_front = []
        for leaves, fragments in groups.items():
            z = torch.cat([v[0] for v in fragments])
            hs = torch.cat([v[1] for v in fragments])
            starts = torch.cat([v[2] for v in fragments])
            nt = z.shape[1]
            if leaves == 1:
                if _settings.get('WRITE', 'indexput') == 'kernel':
                    _importlib.import_module('h3_ascend._vendor.spark.index_kernels').write(z, hs, starts, output)
                else:
                    output[hs[:, None], starts[:, None] * 64 + torch.arange(64, device=x.device)] = z
                continue
            budgets = power_of_two_child_budgets(leaves, (16,), depth)
            caps = tuple((v * 64 for v in budgets))
            l = min(32, nt)
            begin = torch.arange(l, device=x.device) * nt // l
            end = (torch.arange(l, device=x.device) + 1) * nt // l
            mid = begin + (end - begin) // 2
            glob = (z + hs[:, None] * n).contiguous()
            landmark = glob[:, mid]
            centers = source.index_select(0, landmark.reshape(-1)).view(*landmark.shape, 128) if _settings.get('LANDMARK') == 'index_select' else source[landmark]
            weights = (end - begin)[None].expand(z.shape[0], -1)
            dirs = _importlib.import_module('h3_ascend._vendor.spark.proxy_four_native').directions(centers, weights, caps, 2) if _settings.get('PROXY_FOUR') == 'cann' and len(caps) == 4 else _importlib.import_module('h3_ascend._vendor.spark.' + ('proxy_terminal_' + _settings.get('PROXY_TERMINAL'))).directions(centers, weights, caps, 2) if _settings.get('PROXY_TERMINAL') in ('constant', 'sort32', 'staged', 'staged_native', 'staged_int32', 'onchip') and caps == (64, 64) else directions(centers, weights, caps) if _settings.get('PROXY', 'torch') == 'torch' else (_importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_general').directions(centers, weights, caps) if _settings.get('PROXY_BP') == '4' else _importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_split').directions(centers, weights, caps, 2) if _settings.get('PROXY_UNIT') == 'split' else _importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_uniform').directions(centers, weights, caps, 2) if _settings.get('PROXY_RANK') == 'uniform' and caps in ((64, 64), (128, 64, 128, 64)) else _importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_ranklayout').directions(centers, weights, caps, 2) if _settings.get('PROXY_RANK') == 'column' else _importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_max').directions(centers, weights, caps, 2)) if _settings.get('PROXY') == 'hybrid' and len(caps) <= 4 and (centers.shape[0] >= 128) else _importlib.import_module('h3_ascend._vendor.spark.proxy_batch').directions(centers, weights, caps, 2) if _settings.get('PROXY') == 'batch' and len(caps) <= 4 and (centers.shape[0] >= 128) else _importlib.import_module('h3_ascend._vendor.spark.proxy_hybrid_ranklayout').directions(centers, weights, caps, 2) if _settings.get('PROXY_LARGE_BATCH') == '2' and centers.shape[0] >= 128 else _importlib.import_module('h3_ascend._vendor.spark.proxy_fused_column').directions(centers, weights, caps) if _settings.get('PROXY_LARGE_RANK') == 'column' else _importlib.import_module('h3_ascend._vendor.spark.proxy_fused').directions(centers, weights, caps)
            if _settings.get('TERMINAL_FUSED') == 'preunit' and caps == (64, 64) and (score_source is not None):
                _importlib.import_module('h3_ascend._vendor.spark.terminal_preunit_fused').run(score_source, z, dirs, hs, starts, output)
                continue
            root_dense = depth == 0 and _settings.get('ROOT_SCORE_DENSE') == '1' and (score_source is not None) and (precision == 'fp16')
            if root_dense:
                ss = _importlib.import_module('h3_ascend._vendor.spark.scores_dense').root(score_source, z.shape[0], dirs)
            elif precision == 'fp16':
                ss = _importlib.import_module('h3_ascend._vendor.spark.scores_preunit').scores(score_source, glob, dirs) if score_source is not None else kernels.scores(source, glob, dirs)
            elif precision == 'cpu_fp16':
                ss = torch.bmm(unit(source[glob]).half().float(), dirs.half().float().transpose(1, 2))
            else:
                feat = source[glob]
                ss = torch.bmm(unit(feat), dirs.transpose(1, 2))
            if _settings.get('TERMINAL') == 'cann' and caps == (64, 64):
                (_importlib.import_module('h3_ascend._vendor.spark.meta32_native').terminal if meta32 else _importlib.import_module('h3_ascend._vendor.spark.terminal_native').run)(ss[:, :, 0], z, hs, starts, output)
                continue
            mapped = _importlib.import_module('h3_ascend._vendor.spark.small_split_native').split(z, ss, caps) if _settings.get('SMALL_SPLIT') == 'cann' and len(caps) == 4 and (nt <= 512) else split(z, ss, caps, positions=z if root_dense else None)
            offset = 0
            leaf = 0
            for cap, budget in zip(caps, budgets):
                child = mapped[:, offset:offset + cap]
                st = starts + leaf
                if budget == 1:
                    _importlib.import_module('h3_ascend._vendor.spark.index_kernels').write(child, hs, st, output) if _settings.get('WRITE', 'indexput') == 'kernel' else output.index_put_((hs[:, None], st[:, None] * 64 + torch.arange(64, device=x.device)), child)
                else:
                    next_front.append((child, hs, st, budget))
                offset += cap
                leaf += budget
        frontier = next_front
        depth += 1
    if rem:
        output[:, a:] = excluded
    return output.long() if meta32 else output

class Spark:

    def __init__(self, c, solver='npu', bm=64):
        self.c = c
        self.solver = solver
        self.bm = bm
        self.n = c['seq'] - c['G']
        self.m = self.n // 64
        self.active = self.m * 64
        self.w = max(1, round(0.1 * self.m))
        self.profile = {}
        self.metric_indices = hilbert_midpoint_sample_indices((c['F'], c['grid_h'], c['grid_w']), self.m, device='cpu').npu()

    def stamp(self, name, t):
        if _settings.get('PROFILE') == '1':
            torch.npu.synchronize()
            self.profile[name] = (time.perf_counter() - t) * 1000
        return time.perf_counter()

    def reblock(self, q, k):
        t = time.perf_counter()
        h = q.shape[2]
        qv = q[0, self.c['G']:].transpose(0, 1)
        kv = k[0, self.c['G']:].transpose(0, 1)
        if _settings.get('METRIC_SAMPLE') == 'row_select':
            ix = self.metric_indices + self.c['G']
            sampled = torch.cat((q[0].index_select(0, ix).transpose(0, 1), k[0].index_select(0, ix).transpose(0, 1)), 0).float()
        else:
            sampled = torch.cat((qv[:, self.metric_indices], kv[:, self.metric_indices]), 0).float()
        gram = torch.bmm(sampled.transpose(1, 2), sampled) / self.m
        gram = (gram + gram.transpose(1, 2)) * 0.5
        diag = gram.diagonal(dim1=-2, dim2=-1)
        diag.add_(0.001 * diag.mean(-1, keepdim=True))
        t = self.stamp('metric_gram', t)
        factors = (_importlib.import_module('h3_ascend._vendor.spark.cholesky_panel').cholesky(gram.contiguous()) if self.solver == 'panel' else kernels.cholesky(gram.contiguous())) if self.solver != 'cpu' else torch.linalg.cholesky_ex(gram)[0]
        t = self.stamp('metric_factor', t)
        if _settings.get('TREE_GRAPH', '0') == '1':
            if not hasattr(self, 'tree_plan'):
                self.tree_plan = _importlib.import_module('h3_ascend._vendor.spark.tree_graph').PreparedTree(h, self.n, q.device)
            features = self.tree_plan.input
            torch.bmm(kv, factors[:h].to(torch.bfloat16), out=features[:h])
            torch.bmm(qv, factors[h:].to(torch.bfloat16), out=features[h:])
            t = self.stamp('metric_transform', t)
            perm = self.tree_plan()
        else:
            features = torch.cat((torch.bmm(kv, factors[:h].to(torch.bfloat16)), torch.bmm(qv, factors[h:].to(torch.bfloat16))), 0).to(torch.bfloat16) if _settings.get('TRANSFORM', 'native') == 'native' else _importlib.import_module('h3_ascend._vendor.spark.feature_kernels').transform(q, k, factors, self.n, self.c['G'])
            t = self.stamp('metric_transform', t)
            perm = tree(features)
        self.stamp('tree', t)
        return (perm[h:], perm[:h])

    def __call__(self, q, k, v, layer=0):
        t = time.perf_counter()
        h = q.shape[2]
        qi, ki = self.reblock(q, k)
        t = self.stamp('reblock', t)
        pad = 64 if _settings.get('ATTN') in ('cann_joint', 'cann_fused') else 0
        kv_event = None
        if _settings.get('KV_TAIL_INPLACE') == '1':
            qp = kernels.permute(q, qi, self.c['G'], pad=pad)
            kp, vp = _importlib.import_module('h3_ascend._vendor.spark.permute_tail_native').kv(k, v, ki, self.c['G'], (self.m + 63) // 64 * 64)
        elif _settings.get('PERMUTE_NATIVE') == '1':
            qp = kernels.permute(q, qi, self.c['G'], pad=pad)
            kp, vp = _importlib.import_module('h3_ascend._vendor.spark.permute_native').kv(k, v, ki, self.c['G'], pad)
        elif _settings.get('PERMUTE_OVERLAP') == '1':
            kp, vp, kv_event = _importlib.import_module('h3_ascend._vendor.spark.attention_overlap').kv_permute(k, v, ki, self.c['G'], pad)
            qp = kernels.permute(q, qi, self.c['G'], pad=pad)
        elif _settings.get('PERMUTE_FUSED') == 'qkv':
            qp, kp, vp = _importlib.import_module('h3_ascend._vendor.spark.permute_fused').qkv(q, k, v, qi, ki, self.c['G'], pad)
        elif _settings.get('PERMUTE_FUSED') == 'kv':
            qp = kernels.permute(q, qi, self.c['G'], pad=pad)
            kp, vp = _importlib.import_module('h3_ascend._vendor.spark.permute_fused').kv(k, v, ki, self.c['G'], pad)
        else:
            qp = kernels.permute(q, qi, self.c['G'], pad=pad)
            kp = kernels.permute(k, ki, self.c['G'], pad=pad)
            vp = kernels.permute(v, ki, self.c['G'], pad=pad)
        t = self.stamp('permute', t)
        if _settings.get('QPOOL', 'torch') == 'kernel':
            qm, anchor = _importlib.import_module('h3_ascend._vendor.spark.feature_kernels').query_pool(qp, self.m)
        else:
            qm = qp[:, :self.active].view(h, self.m, 64, 128).float().mean(2).to(torch.bfloat16)
            anchor = None
        t = self.stamp('q_pool', t)
        if kv_event is not None:
            torch.npu.current_stream().wait_event(kv_event)
            kp.record_stream(torch.npu.current_stream())
            vp.record_stream(torch.npu.current_stream())
        prepacked = _importlib.import_module('h3_ascend._vendor.spark.attention_overlap').raw_pack(qp, kp, vp, self.m, self.c['seq']) if _settings.get('PREPACK_KV') == '1' else None
        summary_future = _importlib.import_module('h3_ascend._vendor.spark.attention_overlap').summary(anchor, kp, vp, self.m) if _settings.get('SUMMARY_OVERLAP') == '1' and anchor is not None else None
        if _settings.get('KC') == 'cann':
            km = _importlib.import_module('h3_ascend._vendor.spark.preproc_native').centroids(kp, self.m)
        elif _settings.get('KC') == 'split':
            km = _importlib.import_module('h3_ascend._vendor.spark.route_kernels').centroids_split(kp, self.m)
        elif _settings.get('KC', 'official') == 'official':
            km = _importlib.import_module('h3_ascend._vendor.spark.route_kernels').centroids(kp, self.m)
        else:
            km = kp[:, :self.active].reshape(h, self.m, 64, 128).float().mean(2).to(torch.bfloat16)
        t = self.stamp('k_pool', t)
        scores = torch.bmm(qm, km.transpose(1, 2))
        scores = scores if _settings.get('ROUTE_RADIX') == '1' or _settings.get('ROUTE_HYBRID') == '1' else scores.float()
        t = self.stamp('route_scores', t)
        if _settings.get('ROUTE') == 'cann':
            ids, route = _importlib.import_module('h3_ascend._vendor.spark.preproc_native').route(scores, self.w)
        elif _settings.get('ROUTE', 'threshold') == 'legacy':
            ids = scores.topk(self.w, dim=-1, sorted=False).indices.to(torch.int32)
            route = torch.zeros_like(scores, dtype=torch.int8).scatter_(-1, ids.long(), 1)
        else:
            ids, route = _importlib.import_module('h3_ascend._vendor.spark.route_kernels').route(scores, self.w, sort=_settings.get('ROUTE') == 'sort')
        t = self.stamp('route_select', t)
        if _settings.get('PROFILE') == '1':
            self.profile['route'] = sum((self.profile[k] for k in ['q_pool', 'k_pool', 'route_scores', 'route_select']))
        if anchor is None:
            anchor = qp[:, :self.active].float().mean(1).to(torch.bfloat16)
        if summary_future is not None:
            (ak, av, lm), event = summary_future
            torch.npu.current_stream().wait_event(event)
            for tensor in (ak, av, lm):
                tensor.record_stream(torch.npu.current_stream())
        else:
            ak, av, lm = kernels.summaries(anchor, kp, vp, self.m) if _settings.get('SUMMARY', 'cube') == 'cube' else _importlib.import_module('h3_ascend._vendor.spark.summary_vec').summaries(anchor, kp, vp, self.m)
        t = self.stamp('summaries', t)
        out = kernels.attention(qp, kp, vp, ids, route, ak, av, lm, self.bm, **dict(real_tokens=self.c['seq'], **{'prepacked': prepacked} if prepacked is not None else {}) if pad else {})
        t = self.stamp('attention', t)
        result = _importlib.import_module('h3_ascend._vendor.spark.inverse_native').inverse(out, qi, self.c['G']) if _settings.get('INVERSE') == 'cann' else kernels.permute(out, qi, self.c['G'], inverse=True)
        self.stamp('inverse', t)
        return result
