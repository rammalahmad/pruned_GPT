import torch.nn as nn

# set up the initial hooks for all the corresponding layers
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm


def delete_importance_attr(layer: nn.Module):
    if hasattr(layer, "calculated_importance"):
        del layer.calculated_importance
    else:
        raise AttributeError("No importance attribute found in the layer")


def remove_all_forward_hooks(model):
    first_ln = True
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.c_proj._forward_hooks.clear()
            delete_importance_attr(module.c_proj)
        elif isinstance(module, GPT2MLP):
            module.c_fc._forward_hooks.clear()
            delete_importance_attr(module.c_fc)
        elif isinstance(module, LayerNorm) and first_ln:
            module._forward_hooks.clear()
            delete_importance_attr(module)
            first_ln = False
        else:
            continue


def register_all_forward_hooks(model):
    first_ln = True
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            module.c_proj.register_forward_hook(mha_importance_hook)
        elif isinstance(module, GPT2MLP):
            module.c_fc.register_forward_hook(mlp_importance_hook)
        elif isinstance(module, LayerNorm) and first_ln:
            module.register_forward_hook(embedding_importance_hook)
            first_ln = False
        else:
            continue

def mha_importance_hook(module, ins, outs) -> None:
    """Extracts attention output before projection and computes importance per head."""
    # We hook this to the projection layer of the attention module
    
    attn_output = ins[0]  # This is the attention output before projection
    batch_size, seq_len, hidden_dim = attn_output.shape

    parent_module = module._forward_hooks  # Hooks are stored here, so parent is accessible
    for mod in parent_module.values():
        if isinstance(mod, GPT2Attention):
            num_heads = mod.num_heads
            head_dim = mod.head_dim
            break
    else:
        raise ValueError("Parent GPT2Attention module not found")

    # Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
    attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    attn_output = attn_output.norm(p=2, dim=3) # (batch, seq_len, num_heads)
    importance = attn_output.norm(p=2, dim=0).mean(dim=0)

    module.importance_scores = importance  # Store the importance scores


def mlp_importance_hook(module, ins, outs) -> None:
    """calculates the neuron importance for the given layer"""

    # We hook this into the first linear layer of the MLP
    # calculate the importances
    importance = outs.detach().cpu().norm(p=2, dim=0).mean(dim=0)
    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")

    module.calculated_importance = importance

    
    
def embedding_importance_hook(module, ins, outs) -> None:
    # the first block's first processing layer will be the
    # layer norm
    # so we'll just sum up the layer norm outputs after getting them
    # calculate the importances

    importance = outs.detach().cpu().norm(p=2, dim=0).mean(dim=1)
    # print("importance.shape:", importance.shape)
    # print("n_embd: ", outs.size(-1))
    # print("module:", module.__class__.__name__)
    # print("outs.shape:", outs.shape) # probably (B, T, E)

    module.calculated_importance = importance

    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")