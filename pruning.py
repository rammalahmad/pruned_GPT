import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm
import torch
from pruning_utils import compute_pruned_sums, pruned_layer, pruned_layernorm, pruned_embedding, pruned_attention

def prune_model(model, hidden_size, num_heads, embed_size):
    prune_heads(model, num_heads)
    prune_mlp(model, hidden_size)
    prune_embeddings(model, embed_size)

def prune_mlp(model, hidden_size:int) -> None:
    # goal: trim the width of the MLP layers in the transformer blocks
    # hidden_size: the new hidden size of the MLP
    # model = model.to("cpu") can be used for debugging to avoid cuda errors
    # model.to("cpu")
    for module in model.modules():
        if isinstance(module, GPT2MLP):
            importances = module.c_fc.importance_scores
            num_neurons = min(hidden_size, module.c_fc.weight.shape[1])
            idx, _ = importances.argsort(descending=True)[:num_neurons].sort()
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
            top_heads, _ = importances.argsort(descending=True)[:new_num_heads].sort()

            module = pruned_attention(module, top_heads, model.device)

def prune_embeddings(model, new_embed_dim:int) -> None:
    model_blocks = list(model.transformer.h)
    for i, module in enumerate(model_blocks):
        assert new_embed_dim <= module.ln_1.normalized_shape[0], "New embedding dimension is greater than the current embedding dimension"
        idx_ln1, _ = module.ln_1.importance_scores.argsort(descending=True)[:new_embed_dim].sort()
        idx_ln2, _ = module.ln_2.importance_scores.argsort(descending=True)[:new_embed_dim].sort()
        module.ln_1 = pruned_layernorm(module.ln_1, idx_ln1, model.device)
        module.ln_2 = pruned_layernorm(module.ln_2, idx_ln2, model.device)
        if i == 0:
            idx_first_save = idx_ln1.clone() # save it to modify embedding layer later
        if i>0:
            model_blocks[i-1].mlp.c_proj = pruned_layer(module.mlp.c_proj, idx_ln1, model.device, dim=1)
        module.attn.c_attn = pruned_layer(module.attn.c_attn, idx_ln1, model.device, dim=0)
        module.attn.c_proj = pruned_layer(module.attn.c_proj, idx_ln2, model.device, dim=1)
        module.mlp.c_fc = pruned_layer(module.mlp.c_fc, idx_ln2, model.device, dim=0)
        
    idx_lnf, _ = model.transformer.ln_f.importance_scores.argsort(descending=True)[:new_embed_dim].sort()
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
