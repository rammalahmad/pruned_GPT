import torch.nn as nn
import torch

from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm


def delete_importance_attr(layer: nn.Module):
    if hasattr(layer, "importance_scores"):
        del layer.importance_scores
    if hasattr(layer, "importance_buffer"):
        del layer.importance_buffer
    if hasattr(layer, "num_heads"):
        del layer.num_heads


def remove_all_forward_hooks(model):
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.c_proj._forward_hooks.clear()
            delete_importance_attr(module.c_proj)
        elif isinstance(module, GPT2MLP):
            module.c_fc._forward_hooks.clear()
            delete_importance_attr(module.c_fc)
        elif isinstance(module, LayerNorm):
            module._forward_hooks.clear()
            delete_importance_attr(module)


def register_all_forward_hooks(model):
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.c_proj.num_heads = model.config.num_attention_heads
            module.c_proj.importance_buffer = []
            module.c_proj.register_forward_hook(mha_importance_hook)
        elif isinstance(module, GPT2MLP):
            module.c_fc.importance_buffer = []
            module.c_fc.register_forward_hook(mlp_importance_hook)
        elif isinstance(module, LayerNorm):
            module.importance_buffer = []
            module.register_forward_hook(embedding_importance_hook)


def mha_importance_hook(module, ins, outs) -> None:
    """Extracts attention output before projection and accumulates data for importance computation."""
    
    attn_output = ins[0]  # This is the attention output before projection
    batch_size, seq_len, hidden_dim = attn_output.shape

    num_heads = module.num_heads
    assert hidden_dim % num_heads == 0, "Hidden dim is not evenly divisible by num_heads"

    head_dim = hidden_dim // num_heads

    # Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
    attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    attn_output = attn_output.norm(p=2, dim=3) 
    module.importance_buffer.append(attn_output.detach().cpu())


def mlp_importance_hook(module, ins, outs) -> None:
    """Stores outputs for later importance computation."""
    
    module.importance_buffer.append(outs.detach().cpu())


def embedding_importance_hook(module, ins, outs) -> None:
    """Stores embedding layer outputs for later importance computation."""
    
    module.importance_buffer.append(outs.detach().cpu())


def compute_importance_scores(model):
    """Concatenates stored outputs and computes importance scores properly."""
    for module in model.modules():
        if hasattr(module, "importance_buffer") and module.importance_buffer:
            all_outputs = torch.cat(module.importance_buffer, dim=0)  # Concatenate over batch dimension

            # Compute norm-based importance
            importance = all_outputs.norm(p=2, dim=0).mean(dim=0)

            module.importance_scores = importance
            del module.importance_buffer