from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import torch.nn.functional as F


def train_kd(model, teacher_model, tokenized_datasets, batch_size=4, num_epochs = 10, accumulation_steps = 8, lr=1e-5, temperature = 1.0, device='cuda'):
    teacher_model.eval()
    model.train()
    
    dataloader = DataLoader(
        tokenized_datasets,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=lr)  
    

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                teacher_logits = teacher_outputs.logits / temperature

            student_outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            student_logits = student_outputs.logits

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            loss = loss / accumulation_steps

            loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
