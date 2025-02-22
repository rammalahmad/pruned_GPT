import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm
import torch
from utils import compute_pruned_sums, pruned_layer, pruned_layernorm, pruned_embedding, pruned_attention


def prune_mlp(model, mult_factor: float = 4.0) -> None:
    # goal: trim the width of the MLP layers in the transformer blocks
    # mult_factor: the ratio of the input dimension to the input dimension of the MLP layers
    # model = model.to("cpu") can be used for debugging to avoid cuda errors

    for module in model.modules():
        if isinstance(module, GPT2MLP):
            importances = module.c_fc.importance_scores
            num_neurons = min(int(module.c_fc.nx * mult_factor), module.c_fc.weight.shape[1])
            idx = importances.argsort(descending=True)[:num_neurons]
            module.c_fc = pruned_layer(module.c_fc, idx, model.device, dim=1)
            module.c_proj = pruned_layer(module.c_proj, idx, model.device, dim=0)


def prune_heads(model, new_num_heads: int) -> None:
    """
    Prune heads in all GPT2Attention layers.

    Args:
        model: The transformer model (GPT2).
        new_num_heads (int): The number of heads to keep.
    """
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            assert new_num_heads <= module.num_heads, "Number of heads to keep is greater than the current number of heads"
            
            importances = module.c_proj.importance_scores
            top_heads = importances.argsort(descending=True)[:new_num_heads]

            module = pruned_attention(module, top_heads, model.device)

def prune_embeddings(model, new_embed_dim:int) -> None:
    model_blocks = list(model.transformer.h)
    for i, module in enumerate(model_blocks):
        assert new_embed_dim <= module.ln_1.normalized_shape[0], "New embedding dimension is greater than the current embedding dimension"
        idx_ln1 = module.ln_1.importance_scores.argsort(descending=True)[:new_embed_dim]
        idx_ln2 = module.ln_2.importance_scores.argsort(descending=True)[:new_embed_dim]
        module.ln_1 = pruned_layernorm(module.ln_1, idx_ln1, model.device)
        module.ln_2 = pruned_layernorm(module.ln_2, idx_ln2, model.device)
        if i == 0:
            idx_first_save = idx_ln1.clone() # save it to modify embedding layer later
        if i>0:
            model_blocks[i-1].mlp.c_proj = pruned_layer(module.mlp.c_proj, idx_ln1, model.device, dim=1)
        module.attn.c_attn = pruned_layer(module.attn.c_attn, idx_ln1, model.device, dim=0)
        module.attn.c_proj = pruned_layer(module.attn.c_proj, idx_ln2, model.device, dim=1)
        module.mlp.c_fc = pruned_layer(module.mlp.c_fc, idx_ln2, model.device, dim=0)
        
    idx_lnf = model.transformer.ln_f.importance_scores.argsort(descending=True)[:new_embed_dim]
    model.transformer.ln_f = pruned_layernorm(model.transformer.ln_f, idx_lnf, model.device)
    model_blocks[-1].mlp.c_proj = pruned_layer(module.mlp.c_proj, idx_lnf, model.device, dim=1)
    model.transformer.wte = pruned_embedding(model.transformer.wte, idx_first_save, model.device, dim=1)
    model.transformer.wpe = pruned_embedding(model.transformer.wpe, idx_first_save, model.device, dim=1)
    # model.lm_head = pruned_layer(model.lm_head, idx_lnf, model.device, dim=0) # we don't prune the model head since it's tied to the input embeddings
    model.lm_head = nn.Linear(new_embed_dim, model.lm_head.out_features, bias=False)
    model.tie_weights() # tie the weights of the input embeddings and the output projection layer


AVAILABLE_PRUNING_STRATEGIES = {
    "width_head": prune_heads,
    "width_neuron": prune_mlp,
    "width_embedding": prune_embeddings,
}





# def prune_heads(model, new_num_heads:int) -> None:
#     for module in model.modules():
#         if isinstance(module, GPT2Attention):
#             assert new_num_heads <= module.num_heads, "Number of heads to keep is greater than the number of heads in the model"
#             head_size = module.head_dim
#             split_size = module.split_size
            
#             importances = module.c_proj.importance_scores
#             top_heads = importances.argsort(descending=True)[:new_num_heads]
#             pruned_heads = importances.argsort(descending=True)[new_num_heads:]
            
#             def get_full_indices(heads):
#                 return torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in heads])

#             full_indices_keep = get_full_indices(top_heads)
            
#             # Calculate the sum of pruned heads for key, query, and value
#             pruned_sum, pruned_bias_sum = compute_pruned_sums(module, pruned_heads)

#             # Adjust indices for QKV weights
#             index_attn = torch.cat([
#                 full_indices_keep,
#                 full_indices_keep + split_size,  # Key
#                 full_indices_keep + 2 * split_size  # Value
#             ])

#             # Apply pruning
#             pruned_attn_layer = pruned_layer(module.c_attn, index_attn, model.device, dim=1)
#             module.c_attn.weight.data = pruned_sum - (len(pruned_heads) - 1) * pruned_attn_layer.weight.data
#             if module.c_attn.bias is not None:
#                 module.c_attn.bias.data = pruned_bias_sum - (len(pruned_heads) - 1) * pruned_attn_layer.bias.data
#             module.c_proj = pruned_layer(module.c_proj, full_indices_keep, model.device, dim=0)
            

#             # Update the split size and number of heads
#             assert (module.split_size // module.num_heads) * new_num_heads == len(full_indices_keep), "Invalid split size"
#             module.split_size = len(full_indices_keep)
#             module.num_heads = new_num_heads