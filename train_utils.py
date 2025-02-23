from pruning import prune_heads, prune_mlp, prune_embeddings
from hooks import *
import torch
import copy  # For deep copying the model
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

def load_model(model_name, device='auto'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    return model, tokenizer

def trainer_gpt2(model, tokenizer, dataset, output_dir=None,  num_epochs=3, batch_size=4, lr=5e-4):
    """
    HF Trainer for GPT-2 model on the given dataset and returns a dictionary of training & validation losses.
    """
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir, #"test",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        save_steps=0,
        save_total_limit=0,
        fp16=True,
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
                prune_heads(model, num_heads)
                prune_mlp(model, int(hidden_size*embed_size))
                prune_embeddings(model, embed_size)

                # Calculate model size
                model_size = sum(p.numel() for p in model.parameters())

                if param_range[0] <= model_size <= param_range[1]:
                    acceptable_params.append({"num_heads": num_heads, "hidden_size": hidden_size, "embed_size": embed_size, "model_size": model_size})
    if acceptable_params:
        with open("pruning_params.json", "w") as f:
            json.dump(acceptable_params, f, indent=4)

    return acceptable_params


def calibration_pass(model, calib_data):
    """Performs a calibration pass on the model."""
    model.eval()
    register_all_forward_hooks(model)
    with torch.no_grad():
        for batch in calib_data:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
    compute_importance_scores(model)
    

if __name__ == "__main__":
    
    num_heads_options = [8, 10, 12]
    hidden_size_options = [2.5, 3, 3.5, 4]
    embed_size_options = [512, 640, 768]

    param_range = (115_000_000, 135_000_000)

    model_name = "openai-community/gpt2-medium"
    base_model, tokenizer = load_model(model_name)

    acceptable_params = find_acceptable_model_sizes(base_model, tokenizer, num_heads_options, hidden_size_options, embed_size_options, param_range)
    print(acceptable_params)
    print(len(acceptable_params))
    