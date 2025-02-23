from pruning import prune_heads, prune_mlp, prune_embeddings
from datasets import load_dataset
from hooks import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy  # For deep copying the model
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'



from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

def train_gpt2(dataset, model_checkpoint="gpt2", output_dir="./results", num_epochs=3, batch_size=4, lr=5e-5):
    """
    Trains a GPT-2 model on the given dataset and returns a dictionary of training & validation losses.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a padding token by default
    assert tokenizer.padding_side == "right", "The GPT-2 tokenizer must have right padding"
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024
        )

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Load GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure pad token is set

    # Define data collator for padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 does causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        num_train_epochs=num_epochs,
        fp16=True,
        learning_rate=lr,
        gradient_accumulation_steps=2,
        report_to="none"  # Disable logging to external services
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model and return metrics
    train_result = trainer.train()
    metrics = train_result.metrics

    # Evaluate model and get validation loss
    eval_metrics = trainer.evaluate()
    metrics["eval_loss"] = eval_metrics["eval_loss"]

    return metrics



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

    # Load model and tokenizer and do a forward pass
    model_name = "openai-community/gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    acceptable_params = find_acceptable_model_sizes(base_model, tokenizer, num_heads_options, hidden_size_options, embed_size_options, param_range)
    print(acceptable_params)
    print(len(acceptable_params))
    