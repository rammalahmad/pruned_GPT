import torch
from datasets import load_dataset
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def evaluate_perplexity(model, tokenizer, test=test, stride = 1024):
    """
    Evaluates a GPT-2 model's perplexity on a dataset, ignoring padding tokens.
    """
    model.eval()
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)

    return ppl


from datasets import load_dataset

def evaluate_multiple_datasets(model, tokenizer):
    """
    Evaluates perplexity on WikiText-2, WikiText-103, and Penn Treebank (PTB)
    using the existing evaluate_perplexity function.

    Args:
        model: Pretrained GPT-style model.
        tokenizer: Corresponding tokenizer.
        device (str): "cuda" or "cpu".
    
    Returns:
        dict: Perplexity scores for each dataset.
    """

    dataset_map = {
        "WikiText-2": load_dataset("wikitext", "wikitext-2-raw-v1", split="test"),
        "Penn Treebank (PTB)": load_dataset("ptb_text_only", split="test"),
        "WikiText-103": load_dataset("wikitext", "wikitext-103-raw-v1", split="test"),
    }

    results = {}
    stride_map = {
        "WikiText-2": 1024,
        "Penn Treebank (PTB)": 128,
        "WikiText-103": 1024
    }
    for dataset_name, dataset in dataset_map.items():
        print(f"Evaluating {dataset_name}...")
        ppl = evaluate_perplexity(model, tokenizer, test=dataset, stride=stride_map[dataset_name])
        results[dataset_name] = ppl.item()  # Convert tensor to float
        print(f"{dataset_name} Perplexity: {results[dataset_name]}")

    return results
