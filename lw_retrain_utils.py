from pruning import prune_model
from hooks import *
import torch
import copy  # For deep copying the model
import json
import math
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(model_name, device='auto'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == "right" 
    return model, tokenizer

def trainer_gpt2(model, tokenizer, dataset, output_dir="/tmp/trainer",  num_epochs=3, batch_size=4, lr=5e-4):
    """
    HF Trainer for GPT-2 model on the given dataset and returns a dictionary of training & validation losses.
    """
    model.train()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir, #"test",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=30,
        logging_steps=5,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        save_steps=0,
        save_total_limit=0,
        bf16=True,
        fp16=False,
        seed = 3407,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    return trainer


def tokenize_dataset(tokenizer, dataset):
    """Tokenizes the dataset using the tokenizer."""
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    def tokenize_function(examples):
        return {"input_ids": tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024
        )["input_ids"]}

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])
    return tokenized_datasets


def find_acceptable_model_sizes(base_model, tokenizer, num_heads_options, hidden_size_options, embed_size_options, param_range):
    """Finds all acceptable model sizes within the specified parameter range."""

    acceptable_params = []
    
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
                prune_model(model, int(hidden_size*embed_size), num_heads, embed_size)

                # Calculate model size
                model_size = sum(p.numel() for p in model.parameters())

                if param_range[0] <= model_size <= param_range[1]:
                    acceptable_params.append({"num_heads": num_heads, "hidden_size": hidden_size, "embed_size": embed_size, "model_size": model_size})
    if acceptable_params:
        with open("pruning_params.json", "w") as f:
            json.dump(acceptable_params, f, indent=4)

    return acceptable_params


def calibration_pass(model, tokenizer, dataset, sample_size=512, batch_size=16):
    """Performs a calibration pass on the model."""
    data = dataset["train"]
    sampled_data = data.shuffle(seed=42).select(range(sample_size))
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")
    
    tokenized_calib_data = sampled_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_calib_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    calib_data = DataLoader(tokenized_calib_data, batch_size=batch_size)
    
    model.eval()
    register_all_forward_hooks(model)
    with torch.no_grad():
        for batch in tqdm(calib_data):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
    compute_importance_scores(model)
    model.train()
    
    
def evaluate_perplexity(model, tokenizer, stride = 1024):
    """
    Evaluates a GPT-2 model's perplexity on the Wikitext-2 dataset, ignoring padding tokens.
    """
    model.eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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

    print(f"Perplexity on Wikitext-2: {ppl:.2f}")
    return ppl
