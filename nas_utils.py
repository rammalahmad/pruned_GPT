from pruning import prune_heads, prune_mlp, prune_embeddings
from datasets import load_dataset
from hooks import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy  # For deep copying the model
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_acceptable_model_sizes():
    """Finds all acceptable model sizes within the specified parameter range."""

    acceptable_params = []
    
    num_heads_options = [8, 10, 12]
    hidden_size_options = [1536, 2048, 2560, 3072]
    embed_size_options = [512, 640, 768]

    param_range = (115_000_000, 135_000_000)
    
    # Load model and tokenizer and do a forward pass
    model_name = "openai-community/gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    register_all_forward_hooks(base_model)
    
    prompt = "The future is AI"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = base_model(**inputs)
    compute_importance_scores(base_model)


    # Iterate over all pruning configurations
    for num_heads in num_heads_options:
        for hidden_size in hidden_size_options:
            for embed_size in embed_size_options:
                model = copy.deepcopy(base_model)

                # Apply pruning
                prune_heads(model, num_heads)
                prune_mlp(model, hidden_size)
                prune_embeddings(model, embed_size)

                # Calculate model size
                model_size = sum(p.numel() for p in model.parameters())

                if param_range[0] <= model_size <= param_range[1]:
                    acceptable_params.append({"num_heads": num_heads, "hidden_size": hidden_size, "embed_size": embed_size, "model_size": model_size})
    if acceptable_params:
        with open("pruning_params.json", "w") as f:
            json.dump(acceptable_params, f, indent=4)


    return acceptable_params

if __name__ == "__main__":
    acceptable_params = find_acceptable_model_sizes()
    print(acceptable_params)
    print(len(acceptable_params))
    