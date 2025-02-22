from pruning import prune_heads, prune_mlp, prune_embeddings
from datasets import load_dataset
from hooks import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy  # For deep copying the model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load OpenWebText dataset from Hugging Face
# dataset = load_dataset("openwebtext")

num_heads_options = [6, 8, 10, 12]
hidden_size_options = [1536, 2048, 2560, 3072]
embed_size_options = [512, 640, 768]

param_range = (115_000_000, 135_000_000)

def find_acceptable_model_sizes():
    """Finds all acceptable model sizes within the specified parameter range."""

    acceptable_params = []

    # Iterate over all pruning configurations
    for num_heads in num_heads_options:
        for hidden_size in hidden_size_options:
            for embed_size in embed_size_options:
                
                model_name = "openai-community/gpt2-medium"

                # Load model and tokenizer ONCE
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Force float16 instead of BF16
                    device_map="auto"           # Auto-detect the best device
                )
                register_all_forward_hooks(model)

                batch_size = 16
                total_samples = 1024
                num_batches = total_samples // batch_size

                prompt = "The future of AI is"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    for _ in range(num_batches):
                        outputs = model(**inputs)
                
                compute_importance_scores(model)

                # Apply pruning
                prune_heads(model, num_heads)
                prune_mlp(model, hidden_size)
                prune_embeddings(model, embed_size)

                # Calculate model size
                model_size = sum(p.numel() for p in model.parameters())

                if param_range[0] <= model_size <= param_range[1]:
                    acceptable_params.append((num_heads, hidden_size, embed_size, model_size))

    return acceptable_params

if __name__ == "__main__":
    acceptable_params = find_acceptable_model_sizes()
    print(acceptable_params)
    print(len(acceptable_params))
    