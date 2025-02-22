import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm

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
    
def pruned_layer(layer: nn.Module, idx, device, dim=0) -> None:
    num_neurons = idx.size(0)
    if dim == 0:
        new_layer = nn.Linear(num_neurons, layer.out_features, bias=layer.bias is not None).to(device)
        new_layer.weight.data = layer.weight.data[:, idx]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[idx]
    elif dim == 1:
        new_layer = nn.Linear(layer.in_features, num_neurons, bias=layer.bias is not None).to(device)
        new_layer.weight.data = layer.weight.data[idx, :]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[idx]
    else:
        raise ValueError("Invalid dimension")
    return new_layer

def pruned_layernorm(layer: LayerNorm, idx, device) -> LayerNorm:
    num_neurons = idx.size(0)
    new_layer = LayerNorm(num_neurons, eps=layer.eps, elementwise_affine=layer.elementwise_affine).to(device)

    # Copy weights and bias if affine transformation is used
    if layer.elementwise_affine:
        new_layer.weight.data = layer.weight.data[idx].clone()
        new_layer.bias.data = layer.bias.data[idx].clone()

    return new_layer

def pruned_embedding(layer: nn.Embedding, idx, device, dim=0) -> nn.Embedding:
    """
    Returns a pruned Embedding layer with only selected indices.

    Args:
        layer (nn.Embedding): The original embedding layer.
        idx (torch.Tensor): Indices of tokens (dim=0) or features (dim=1) to keep.
        device (torch.device): The device for the new layer.
        dim (int): Dimension to prune (0 = vocab pruning, 1 = feature pruning).

    Returns:
        nn.Embedding: The pruned embedding layer.
    """
    if dim == 0:
        num_embeddings = idx.size(0)
        new_layer = nn.Embedding(num_embeddings, layer.embedding_dim, padding_idx=layer.padding_idx).to(device)
        new_layer.weight.data = layer.weight.data[idx, :].clone()
    elif dim == 1:
        embedding_dim = idx.size(0)
        new_layer = nn.Embedding(layer.num_embeddings, embedding_dim, padding_idx=layer.padding_idx).to(device)
        new_layer.weight.data = layer.weight.data[:, idx].clone()
    else:
        raise ValueError("Invalid dimension for pruning Embedding layer")

    return new_layer
  
    
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


def pruned_attention(attn_layer, top_heads, model_device):
    """
    Prune an attention layer by keeping only the top_heads.
    
    Args:
        attn_layer (GPT2Attention): The original attention layer.
        top_heads (torch.Tensor): Indices of heads to keep.
        model_device (torch.device): The device for the new layer.
    
    Returns:
        Pruned GPT2Attention layer
    """
    head_size = attn_layer.head_dim
    split_size = attn_layer.split_size

    def get_full_indices(heads):
        return torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in heads])

    full_indices_keep = get_full_indices(top_heads)

    # Calculate the sum of pruned heads for key, query, and value
    pruned_heads = torch.tensor([h for h in range(attn_layer.num_heads) if h not in top_heads], device=model_device)
    pruned_sum, pruned_bias_sum = compute_pruned_sums(attn_layer, pruned_heads)

    # Adjust indices for QKV weights
    index_attn = torch.cat([
        full_indices_keep,
        full_indices_keep + split_size,  # Key
        full_indices_keep + 2 * split_size  # Value
    ])

    # Apply pruning
    pruned_attn_layer = pruned_layer(attn_layer.c_attn, index_attn, model_device, dim=1)
    attn_layer.c_attn.weight.data = pruned_sum - (len(pruned_heads) - 1) * pruned_attn_layer.weight.data
    if attn_layer.c_attn.bias is not None:
        attn_layer.c_attn.bias.data = pruned_bias_sum - (len(pruned_heads) - 1) * pruned_attn_layer.bias.data
    attn_layer.c_proj = pruned_layer(attn_layer.c_proj, full_indices_keep, model_device, dim=0)

    # Update the split size and number of heads
    attn_layer.split_size = len(full_indices_keep)
    attn_layer.num_heads = len(top_heads)

    return attn_layer
