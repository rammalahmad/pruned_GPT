import torch

def model_size(model):
    """Computes the number of parameters in a model."""
    num_params = 0
    sd = model.state_dict()

    for k, v in sd.items():
        # Skip counting the output projection weights since it is tied to the input embeddings
        if k == "lm_head.weight":
            continue
        num_params += v.numel()

    print(f"Corrected Total Parameters: {num_params:,}")
    
    
def compute_pruned_sums(module, pruned_heads: torch.Tensor) -> torch.Tensor:
    """Computes the sum of pruned heads for QKV weights and bias."""
    head_size = module.head_dim
    num_heads = module.num_heads   
    W_q, W_k, W_v = module.c_attn.weight.data.chunk(3, dim=0)
    W_q = W_q.view(num_heads, head_size, -1)
    W_k = W_k.view(num_heads, head_size, -1)
    W_v = W_v.view(num_heads, head_size, -1)
    pruned_q = W_q[pruned_heads].sum(dim=0)
    pruned_k = W_k[pruned_heads].sum(dim=0)
    pruned_v = W_v[pruned_heads].sum(dim=0)
    pruned_sum = torch.cat([pruned_q.repeat(num_heads-len(pruned_heads), 1),
                            pruned_k.repeat(num_heads-len(pruned_heads), 1),
                            pruned_v.repeat(num_heads-len(pruned_heads), 1)], dim=0)
    
    if module.c_attn.bias is not None:
        b_q, b_k, b_v = module.c_attn.bias.data.chunk(3, dim=0)
        b_q = b_q.view(num_heads, head_size)
        b_k = b_k.view(num_heads, head_size)
        b_v = b_v.view(num_heads, head_size)
        pruned_b_q = b_q[pruned_heads].sum(dim=0)
        pruned_b_k = b_k[pruned_heads].sum(dim=0)
        pruned_b_v = b_v[pruned_heads].sum(dim=0)
        pruned_bias_sum = torch.cat([pruned_b_q.repeat(num_heads-len(pruned_heads)),
                                     pruned_b_k.repeat(num_heads-len(pruned_heads)),
                                     pruned_b_v.repeat(num_heads-len(pruned_heads))])
    else:
        pruned_bias_sum = None
    
    return pruned_sum, pruned_bias_sum