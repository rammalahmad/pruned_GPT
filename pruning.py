import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm
import torch
from .utils import compute_pruned_sums

def prune_mlp(model, mult_factor: float = 4.0) -> None:
    # goal: trim the width of the MLP layers in the transformer blocks
    # mult_factor: the ratio of the input dimension to the input dimension of the MLP layers

    for module in model.modules():
        if isinstance(module, GPT2MLP):
            importances = module.c_fc.importance_scores
            num_neurons = int(module.c_fc.in_features * mult_factor)
            idx = importances.argsort(descending=True)[:num_neurons]
            module.c_fc = pruned_layer(module.c_fc, idx, model.device, dim=1)
            module.c_proj = pruned_layer(module.c_proj, idx, model.device, dim=0)

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


def prune_heads(model, new_num_heads:int) -> None:
    for module in model.modules():
        if isinstance(module, GPT2Attention):
            assert new_num_heads <= module.num_heads, "Number of heads to keep is greater than the number of heads in the model"
            head_size = module.head_dim
            split_size = module.split_size
            
            importances = module.c_proj.importance_scores
            top_heads = importances.argsort(descending=True)[:new_num_heads]
            pruned_heads = importances.argsort(descending=True)[new_num_heads:]
            
            def get_full_indices(heads):
                return torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in heads])

            full_indices_keep = get_full_indices(top_heads)
            
            # Calculate the sum of pruned heads for key, query, and value
            pruned_sum, pruned_bias_sum = compute_pruned_sums(module, pruned_heads)

            # Adjust indices for QKV weights
            index_attn = torch.cat([
                full_indices_keep,
                full_indices_keep + split_size,  # Key
                full_indices_keep + 2 * split_size  # Value
            ])

            # Apply pruning
            pruned_attn_layer = pruned_layer(module.c_attn, index_attn, model.device, dim=1)
            module.c_attn.weight.data = pruned_sum - (len(pruned_heads) - 1) * pruned_attn_layer.weight.data
            if module.c_attn.bias is not None:
                module.c_attn.bias.data = pruned_bias_sum - (len(pruned_heads) - 1) * pruned_attn_layer.bias.data
            module.c_proj = pruned_layer(module.c_proj, full_indices_keep, model.device, dim=0)
            

            # Update the split size and number of heads
            assert (module.split_size // module.num_heads) * new_num_heads == len(full_indices_keep), "Invalid split size"
            module.split_size = len(full_indices_keep)
            module.num_heads = new_num_heads

def prune_embeddings(model, new_embed_dim:int) -> None:
    model_blocks = list(model.transformer.h)
    for i, module in enumerate(model_blocks):
        assert new_embed_dim <= module.ln1.in_features, "New embedding dimension is greater than the current embedding dimension"
        idx_ln1 = module.ln1.importance_scores.argsort(descending=True)[:new_embed_dim]
        idx_ln2 = module.ln2.importance_scores.argsort(descending=True)[:new_embed_dim]
        module.ln1 = pruned_layernorm(module.ln1, idx_ln1, model.device)
        module.ln2 = pruned_layernorm(module.ln2, idx_ln2, model.device)
        if i == 0:
            embedding_first_save = idx_ln1.clone() # save it to modify embedding layer later
        if i>0:
            model_blocks[i-1].mlp.c_proj = pruned_layer(module.mlp.c_proj, idx_ln1, model.device, dim=1)
        module.attn.c_attn = pruned_layer(module.attn.c_attn, idx_ln1, model.device, dim=0)
        module.attn.c_proj = pruned_layer(module.attn.c_proj, idx_ln2, model.device, dim=1)
        module.mlp.c_fc = pruned_layer(module.mlp.c_fc, idx_ln2, model.device, dim=0)
        
    idx_lnf = model.ln_f.importance_scores.argsort(descending=True)[:new_embed_dim]
    model.ln_f = pruned_layernorm(model.ln_f, idx_lnf, model.device)
    model_blocks[-1].mlp.c_proj = pruned_layer(module.mlp.c_proj, idx_lnf, model.device, dim=1)
    model.lm_head = pruned_layer(model.lm_head, idx_lnf, model.device, dim=0)

            

def prune_embeddings(model, ratio=0.2) -> None:
    # goal: trim the embedding dimension of the weight matrices in MLP, MHA, and LayerNorm layers.
    importances = model.blocks[0].ln1.calculated_importance
    num_dense_embd = int((1 - ratio) * model.n_embd)
    idx = importances.argsort(descending=True)[:num_dense_embd]



    for module in model.modules():
        if isinstance(module, Block):
            # start with pruning the MLP layers
            importances = module.ln1.calculated_importance

            dense1 = module.ffwd.net[0]  # weights.shape = (emb, 4 * emb)
            dense2 = module.ffwd.net[2]  # weights.shape = (4 * emb, emb)

            module.ffwd.net[0] = nn.Linear(num_dense_embd, dense1.out_features).to(
                model.device
            )  # weights.shape = (num_dense_embd, dense1.in_features)
            module.ffwd.net[2] = nn.Linear(dense2.in_features, num_dense_embd).to(
                model.device
            )  # weights.shape = (dense2.out_features = emb)

            module.ffwd.net[0].weight.data = dense1.weight.data[:, idx]
            module.ffwd.net[0].bias.data = dense1.bias.data
            module.ffwd.net[2].weight.data = dense2.weight.data[idx, :]
            module.ffwd.net[2].bias.data = dense2.bias.data[idx]

            # now the multi-head attention
            for head in module.sa.heads:
                # key,value,query weight shape: (head_size, n_embd) # n_embd
                k, v, q = head.key, head.value, head.query

                head.key = nn.Linear(num_dense_embd, k.out_features, bias=False).to(
                    model.device
                )
                head.value = nn.Linear(num_dense_embd, v.out_features, bias=False).to(
                    model.device
                )
                head.query = nn.Linear(num_dense_embd, q.out_features, bias=False).to(
                    model.device
                )

                head.key.weight.data = k.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)
                head.value.weight.data = v.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)
                head.query.weight.data = q.weight.data[
                    :, idx
                ]  # (head_size, num_dense_embd)

                head.key.calculated_importance = k.calculated_importance
                head.value.calculated_importance = v.calculated_importance
                head.query.calculated_importance = q.calculated_importance

            ln1 = module.ln1
            ln2 = module.ln2

            module.ln1 = nn.LayerNorm(num_dense_embd).to(model.device)
            module.ln1.weight.data = ln1.weight.data[idx]
            module.ln1.bias.data = ln1.bias.data[idx]

            module.ln2 = nn.LayerNorm(num_dense_embd).to(model.device)
            module.ln2.weight.data = ln2.weight.data[idx]
            module.ln2.bias.data = ln2.bias.data[idx]

            proj = module.sa.proj
            module.sa.proj = nn.Linear(proj.in_features, num_dense_embd).to(
                model.device
            )
            module.sa.proj.weight.data = proj.weight.data[
                idx, :
            ]  # (num_dense_embd, n_embd)
            module.sa.proj.bias.data = proj.bias.data[idx]

            module.sa.proj.calculated_importance = proj.calculated_importance

    temb_table = model.token_embedding_table
    pemb_table = model.position_embedding_table

    model.token_embedding_table = nn.Embedding(model.vocab_size, num_dense_embd).to(
        model.device
        ) # type: ignore
    model.position_embedding_table = nn.Embedding(model.block_size, num_dense_embd).to(
        model.device
    )

    model.token_embedding_table.weight.data = temb_table.weight.data[:, idx]
    model.position_embedding_table.weight.data = pemb_table.weight.data[:, idx]

    lnf = model.ln_f
    ln_head = model.ln_head

    model.ln_f = nn.LayerNorm(num_dense_embd).to(model.device)
    model.ln_head = nn.Linear(num_dense_embd, ln_head.out_features).to(model.device)

    model.ln_f.weight.data = lnf.weight.data[idx]
    model.ln_f.bias.data = lnf.bias.data[idx]
    model.ln_head.weight.data = ln_head.weight.data[
        :, idx
    ]  # weight.shape = (vocab_size, embd)
    model.ln_head.bias.data = ln_head.bias.data



class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


AVAILABLE_PRUNING_STRATEGIES = {
    "width_head": prune_heads,
    "width_neuron": prune_mlp,
    "width_embedding": prune_embeddings,
}


