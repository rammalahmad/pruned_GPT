def model_size(model):
    """Computes the number of parameters in a model."""
    num_params = 0
    sd = model.state_dict()

    for k, v in sd.items():
        # Skip counting the output projection weights since it is tied to the input embeddings
        if k == "lm_head.weight":
            continue
        num_params += v.numel()

    print(f"Corrected Total Parameters: {num_params:,}")