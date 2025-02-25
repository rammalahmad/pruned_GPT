# **Pruned GPT**  
**Ahmad Rammal**  
**February 25, 2025**  

![Transformer Diagram](saved_plots/transformers_diagram.png)  

## **Overview**  
This repository provides an **unofficial implementation** of the paper:  
[**Compact Language Models via Pruning and Knowledge Distillation**](https://arxiv.org/pdf/2407.14679) by **NVIDIA**.  

We implement **width and depth pruning** for the **GPT-2 model**, optimizing layers, dimensions, attention heads, and MLP hidden sizes. Additionally, we apply **knowledge distillation** to retain model performance post-pruning.  

Our approach includes **neural architecture search** to evaluate different pruning strategies:  
- **Width pruning:** Reducing model dimensions.  
- **Depth pruning:** Removing layers.  
- **Width + Depth pruning:** A combined approach for optimal compression.  

### **Results**  
The table below compares the perplexity (PPL) scores on WikiText-2, Lambada, and PTB datasets, along with validation loss after lightweight retraining.  

| Model | Size (M) | WikiText-2 (PPL) | Lambada (PPL) | PTB (PPL) | KD Val Loss |
|--------|--------|----------------|--------------|----------|-------------|
| **GPT-2 S** | 124 | 29.41 | 65.85 | 58.41 | -- |
| **GPT-2 M** | 355 | 22.76 | 47.33 | 42.97 | -- |
| **Depth + Width** | 264 | 34.35 | **50.38** | **52.53** | **10.34** |
| **Depth** | 255 | **32.29** | 51.05 | 54.47 | 11.03 |
| **Width** | 254 | 39.47 | 58.66 | 57.32 | 16.25 |

## **Repository Structure**  
**Core Modules**  
- `hooks.py` – Pruning hooks.  
- `pruning.py` & `pruning_utils.py` – Core pruning logic and utilities.  
- `knowledge_distillation.py` – Implements knowledge distillation.  
- `lw_retrain_utils.py` – Lightweight retraining and architecture search tools.  
- `evaluation.py` – Functions for model evaluation.  

**Jupyter Notebooks**  
- `lw_retraining.ipynb` – Runs neural architecture search and retraining.  
- `depth_pruning.ipynb` – Performs depth pruning, retraining, and evaluation.  
- `width_pruning.ipynb` – Performs width pruning, retraining, and evaluation.  
- `width+depth_pruning.ipynb` – Combines width and depth pruning.  

## **Implementation Details**  
**Tech Stack:**  
- **Framework:** PyTorch, Hugging Face Transformers  
- **Python:** 3.10.16  

**Hardware Used:**  
- **GPU:** NVIDIA RTX A5000 (24.5GB VRAM)  

---

This version enhances readability, keeps key points concise, and improves the visual structure. Let me know if you need further tweaks! 🚀