"""Audited process topology and weight replication; no forward math changes."""
import hashlib


def replicated_qkv_shape(shape, hidden_size, total_heads, head_dim):
    """Validate logical full weights in native or transposed INT8 storage."""
    output_size=3*total_heads*head_dim
    assert tuple(shape) in ((hidden_size,output_size),(output_size,hidden_size)), (
        tuple(shape),hidden_size,output_size)
    return dict(logical_input_size=hidden_size,logical_output_size=output_size,
                stored_transposed=tuple(shape)==(hidden_size,output_size))

def metadata(model):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        get_ulysses_parallel_rank,
        get_ulysses_parallel_world_size,
    )
    tp=get_tensor_model_parallel_world_size()
    sp=get_sequence_parallel_world_size()
    up=get_ulysses_parallel_world_size()
    assert tp==1 and sp==up and sp>1,(tp,sp,up)
    assert all(b.attn.num_heads==b.attn.total_num_heads==56 for b in model.blocks)
    result=dict(tp_rank=get_tensor_model_parallel_rank(),tp_world_size=tp,
      sp_rank=get_sequence_parallel_rank(),sp_world_size=sp,
      ulysses_rank=get_ulysses_parallel_rank(),ulysses_world_size=up,
      projection_heads_per_rank=56,attention_heads_after_all_to_all=56//up,
      model_weights_replicated=True)
    samples=[]
    for index in (0,24,49):
        weight=model.blocks[index].attn.qkv_proj.weight
        layout=replicated_qkv_shape(weight.shape,model.arch.hidden_size,
          model.arch.num_attention_heads,model.arch.attention_head_dim)
        payload=weight.detach()[:8,:1024].contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()
        samples.append(dict(layer=index,shape=list(weight.shape),dtype=str(weight.dtype),
          **layout,sample_sha256=hashlib.sha256(payload).hexdigest()))
    result['replicated_qkv_weight_samples']=samples
    return result
