import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm
from transformers.pytorch_utils import Conv1D

def model_size(model):
    """Computes the number of parameters in a model."""
    return sum(t.numel() for t in model.parameters())
    
def pruned_layer(layer: nn.Module, idx, device, dim=0) -> nn.Module:
    """
    Prunes a given layer and replaces it with the appropriate type:
    - If the original layer is `Conv1D`, the new layer is `Conv1D`.
    - If the original layer is `nn.Linear`, the new layer is `nn.Linear`.

    Args:
        layer (nn.Module): The original layer (Conv1D or Linear).
        idx (torch.Tensor): Indices to keep.
        device (torch.device): The device for the new layer.
        dim (int): Dimension to prune (0 = input dim, 1 = output dim).

    Returns:
        nn.Module: The pruned layer (either Conv1D or Linear).
    """
    num_neurons = idx.size(0)
    in_features, out_features = layer.weight.data.shape
    
    # Ensure dtype consistency
    dtype = layer.weight.dtype  

    # Check if the layer is Conv1D
    is_conv1d = isinstance(layer, Conv1D)

    if dim == 0:  # Prune input dimension
        if is_conv1d:
            new_layer = Conv1D(out_features, num_neurons).to(device, dtype=dtype)
            new_layer.weight.data = layer.weight.data[idx, :].clone().to(dtype)  # No need to transpose
        else:  # nn.Linear case
            new_layer = nn.Linear(num_neurons, out_features, bias=layer.bias is not None).to(device, dtype=dtype)
            new_layer.weight.data = layer.weight.data[idx, :].clone().to(dtype).T  # Transpose needed for Linear

    elif dim == 1:  # Prune output dimension
        if is_conv1d:
            new_layer = Conv1D(num_neurons, in_features).to(device, dtype=dtype)
            new_layer.weight.data = layer.weight.data[:, idx].clone().to(dtype)  # No need to transpose
        else:  # nn.Linear case
            new_layer = nn.Linear(in_features, num_neurons, bias=layer.bias is not None).to(device, dtype=dtype)
            new_layer.weight.data = layer.weight.data[:, idx].clone().to(dtype).T  # Transpose needed for Linear
        
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[idx].clone().to(dtype)

    else:
        raise ValueError("Invalid dimension for pruning")

    # Preserve importance_scores if they exist
    if hasattr(layer, "importance_scores"):
        new_layer.importance_scores = layer.importance_scores.clone()
    new_layer.weight.requires_grad = True
    if new_layer.bias is not None:
        new_layer.bias.requires_grad = True

    return new_layer


def pruned_layernorm(layer: LayerNorm, idx, device) -> LayerNorm:
    num_neurons = idx.size(0)
    
    # Ensure dtype consistency
    dtype = layer.weight.dtype if layer.elementwise_affine else torch.float32  # Default to float32 if no affine transformation
    new_layer = LayerNorm(num_neurons, eps=layer.eps, elementwise_affine=layer.elementwise_affine).to(device, dtype=dtype)

    # Copy weights and bias if affine transformation is used
    if layer.elementwise_affine:
        new_layer.weight.data = layer.weight.data[idx].to(dtype)
        new_layer.bias.data = layer.bias.data[idx].to(dtype)
    if hasattr(layer, "importance_scores"):
        new_layer.importance_scores = layer.importance_scores
    new_layer.weight.requires_grad = True
    if new_layer.bias is not None:
        new_layer.bias.requires_grad = True

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
    dtype = layer.weight.dtype  # Ensure dtype consistency
    
    if dim == 0:  # Prune vocabulary (keep selected token indices)
        num_embeddings = idx.size(0)
        new_layer = nn.Embedding(num_embeddings, layer.embedding_dim, padding_idx=layer.padding_idx).to(device, dtype=dtype)
        new_layer.weight.data = layer.weight.data[idx, :].clone().to(dtype)

    elif dim == 1:  # Prune feature dimension (reduce embedding size)
        embedding_dim = idx.size(0)
        new_layer = nn.Embedding(layer.num_embeddings, embedding_dim, padding_idx=layer.padding_idx).to(device, dtype=dtype)
        new_layer.weight.data = layer.weight.data[:, idx].clone().to(dtype)

    else:
        raise ValueError("Invalid dimension for pruning Embedding layer")
    
    new_layer.weight.requires_grad = True

    return new_layer

  
    
def compute_pruned_sums(module, pruned_heads: torch.Tensor) -> torch.Tensor:
    """Computes the sum of pruned heads for QKV weights and bias."""
    dtype = module.c_attn.weight.dtype
    head_size = module.head_dim
    num_heads = module.num_heads
    embed_dim = module.c_attn.nx
    remaining_heads = num_heads - len(pruned_heads)
    
    W_q, W_k, W_v = module.c_attn.weight.data.chunk(3, dim=1)
    W_q = W_q.view(embed_dim, num_heads, head_size).transpose(0, 1)  # (num_heads, embed_dim, head_size)
    W_k = W_k.view(embed_dim, num_heads, head_size).transpose(0, 1)
    W_v = W_v.view(embed_dim, num_heads, head_size).transpose(0, 1)

    # Compute sum over the pruned heads
    pruned_q = W_q[pruned_heads, :, :].sum(dim=0)  # Sum over pruned heads (embed_dim, head_size)
    pruned_k = W_k[pruned_heads, :, :].sum(dim=0)
    pruned_v = W_v[pruned_heads, :, :].sum(dim=0)
    pruned_q = pruned_q.unsqueeze(1).repeat(1, remaining_heads, 1)  # (embed_dim, remaining_heads, head_size)
    pruned_k = pruned_k.unsqueeze(1).repeat(1, remaining_heads, 1)
    pruned_v = pruned_v.unsqueeze(1).repeat(1, remaining_heads, 1)

    # Reshape back to (embed_dim, remaining_heads * head_size)
    pruned_q = pruned_q.view(embed_dim, remaining_heads * head_size)  # (embed_dim, remaining_heads * head_size)
    pruned_k = pruned_k.view(embed_dim, remaining_heads * head_size)
    pruned_v = pruned_v.view(embed_dim, remaining_heads * head_size)

    # Concatenate the QKV components to match attention structure
    pruned_sum = torch.cat([pruned_q, pruned_k, pruned_v], dim=1).to(dtype)
    
    if module.c_attn.bias is not None:
        # Split bias into Q, K, V
        b_q, b_k, b_v = module.c_attn.bias.data.chunk(3, dim=0)  # (3 * embed_dim,)

        b_q = b_q.view(num_heads, head_size)  # (num_heads, head_size)
        b_k = b_k.view(num_heads, head_size)
        b_v = b_v.view(num_heads, head_size)

        # Sum over pruned heads
        pruned_b_q = b_q[pruned_heads, :].sum(dim=0)  # (head_size,)
        pruned_b_k = b_k[pruned_heads, :].sum(dim=0)
        pruned_b_v = b_v[pruned_heads, :].sum(dim=0)

        # ✅ Correctly repeat pruned bias across remaining heads
        pruned_b_q = pruned_b_q.unsqueeze(0).repeat(remaining_heads, 1)  # (remaining_heads, head_size)
        pruned_b_k = pruned_b_k.unsqueeze(0).repeat(remaining_heads, 1)
        pruned_b_v = pruned_b_v.unsqueeze(0).repeat(remaining_heads, 1)

        # Flatten and concatenate biases
        pruned_bias_sum = torch.cat([pruned_b_q.flatten(), pruned_b_k.flatten(), pruned_b_v.flatten()], dim=0).to(dtype)
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

    full_indices_keep = torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in top_heads])

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
    attn_layer.c_attn = Conv1D(pruned_sum.shape[1], pruned_sum.shape[0]).to(model_device, dtype=attn_layer.c_attn.weight.dtype)
    attn_layer.c_attn.weight.data = pruned_sum - (len(pruned_heads) - 1) * pruned_attn_layer.weight.data
    attn_layer.c_attn.weight.requires_grad = True
    if attn_layer.c_attn.bias is not None:
        attn_layer.c_attn.bias.data = pruned_bias_sum - (len(pruned_heads) - 1) * pruned_attn_layer.bias.data
        attn_layer.c_attn.bias.requires_grad = True
    attn_layer.c_proj = pruned_layer(attn_layer.c_proj, full_indices_keep, model_device, dim=0)

    # Update the split size and number of heads
    attn_layer.split_size = len(full_indices_keep)
    attn_layer.num_heads = len(top_heads)

    return attn_layer
