import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm  # Progress bar

def train_kd(model, teacher_model, tokenized_datasets, validation_dataset, 
             batch_size=4, num_epochs=2, accumulation_steps=8, lr=1e-5, 
             temperature=1.0, device='cuda', log_interval=10, val_interval=500):
    
    teacher_model.eval()
    model.train()
    
    train_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)  

    # Tracking loss for training and validation
    train_losses = []
    val_losses = []
    steps = []

    global_step = 0
    steps_since_val = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        print(f"\n🔄 Epoch {epoch+1}/{num_epochs} - Training...")
        train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", leave=False)

        for batch_idx, batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                teacher_logits = teacher_outputs.logits / temperature

            student_outputs = model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
            student_logits = student_outputs.logits

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss = loss / accumulation_steps

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

            # Save loss every `log_interval` steps
            if (batch_idx + 1) % log_interval == 0:
                avg_train_loss = total_train_loss / (batch_idx + 1)
                train_losses.append(avg_train_loss)
                steps.append(global_step)
                train_bar.set_description(f"Training Loss: {avg_train_loss:.4f}")  # Update tqdm description

            global_step += 1
            steps_since_val += 1

            # 🔍 Run validation every `val_interval` steps
            if steps_since_val >= val_interval:
                avg_val_loss = validate_kd(model, teacher_model, val_dataloader, temperature, device)
                print(f"📉 Step {global_step}: Validation Loss = {avg_val_loss:.4f}")
                val_losses.append(avg_val_loss)
                steps_since_val = 0  # Reset validation counter

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"✅ Epoch {epoch+1}: Average Training Loss = {avg_train_loss:.4f}")

        # Final validation at the end of each epoch
        avg_val_loss = validate_kd(model, teacher_model, val_dataloader, temperature, device)
        print(f"📉 Epoch {epoch+1}: Final Validation Loss = {avg_val_loss:.4f}")
        val_losses.append(avg_val_loss)

    return steps, train_losses, val_losses


def validate_kd(model, teacher_model, val_dataloader, temperature, device):
    """Runs validation and returns the average validation loss."""
    model.eval()
    total_val_loss = 0

    val_bar = tqdm(val_dataloader, total=len(val_dataloader), desc="Validation", leave=False)

    with torch.no_grad():
        for batch in val_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits / temperature

            student_outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            student_logits = student_outputs.logits

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

            total_val_loss += loss.item()
            val_bar.set_description(f"Validation Loss: {total_val_loss / (val_bar.n + 1):.4f}")

    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss
