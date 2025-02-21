import torch.nn as nn

# set up the initial hooks for all the corresponding layers
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from torch.nn.modules.normalization import LayerNorm


def delete_importance_attr(layer: nn.Module):
    if hasattr(layer, "importance_scores"):
        del layer.importance_scores
    # else:
    #     raise AttributeError("No importance attribute found in the layer")
    if hasattr(layer, "num_heads"):
        del layer.num_heads


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
            module.c_proj.num_heads = model.config.num_attention_heads
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

    num_heads = module.num_heads
    assert hidden_dim % num_heads == 0, "Hidden dim is not evenly divisible by num_heads"

    head_dim = hidden_dim // num_heads

    # Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
    attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
    attn_output = attn_output.norm(p=2, dim=3) # (batch, seq_len, num_heads)
    importance = attn_output.detach().cpu().norm(p=2, dim=0).mean(dim=0)

    module.importance_scores = importance  # Store the importance scores


def mlp_importance_hook(module, ins, outs) -> None:
    """calculates the neuron importance for the given layer"""

    # We hook this into the first linear layer of the MLP
    # calculate the importances
    importance = outs.detach().cpu().norm(p=2, dim=0).mean(dim=0)
    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")

    module.importance_scores = importance

    
    
def embedding_importance_hook(module, ins, outs) -> None:
    # We hook this into the first layer normalization layer of the model
    importance = outs.detach()
    importance = importance.cpu().norm(p=2, dim=0).mean(dim=0)
    # print("importance.shape:", importance.shape)
    # print("n_embd: ", outs.size(-1))
    # print("module:", module.__class__.__name__)
    # print("outs.shape:", outs.shape) # probably (B, T, E)

    module.importance_scores = importance

    # print(f"{module.__class__.__name__} importance.shape: {importance.shape}")
    
    
    
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    device = 'cuda'
    model_name = "openai-community/gpt2-medium"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Force float16 instead of BF16
        device_map="auto"           # Auto-detect the best device
    )
    
    register_all_forward_hooks(model)
    
    # Sample prompt
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Decode output
    generated_tokens = outputs.logits.argmax(dim=-1)  # Get the most likely token
    generated_text = tokenizer.decode(generated_tokens[0])

    print("Generated text:", generated_text)

    # Print importance scores for each registered module
    for name, module in model.named_modules():
        if hasattr(module, "importance_scores"):
            print(f"Layer {name} importance scores:", module.importance_scores)