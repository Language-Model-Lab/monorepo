import torch
from kernel_backend.errors import InvalidLayerError, InvalidHeadError

def add_head_scaling(model, head_indices: list[int], layer_idx: int, scale: float=0.0):
    #Number of hidden layers
    num_layers = model.config.num_hidden_layers
    #Number of Query attention heads
    num_heads = model.config.num_attention_heads

    if not 0 <= layer_idx < num_layers:
        raise InvalidLayerError(
            f"Layer index {layer_idx} is invalid. "
            f"Valid range is 0-{num_layers - 1}."
            )
    for head_idx in head_indices:
        if not 0 <= head_idx < num_heads:
            raise InvalidHeadError(
                f"Attention head {head_idx} is invalid. "
                f"Valid range is 0-{num_heads - 1}."
                )

    #Projection dimension per attention head
    head_dim = model.model.layers[layer_idx].self_attn.head_dim

    # Load final attention module
    attention_dense = model.model.layers[layer_idx].self_attn.o_proj

    def head_scaling_hook(module, inputs):
        x = inputs[0].clone()

        for head_idx in head_indices:
            #compute indices corresponding to specific head
            start = head_idx * head_dim
            end = start + head_dim

            #Select indices in the head outputs vector corresponding to chosen head and scale them
            x[..., start:end] *= scale

        return (x,) + inputs[1:]

    #Note this is a pre-hook because we want to intervene components before the o-projection forward pass
    handle = attention_dense.register_forward_pre_hook(head_scaling_hook)

    return handle, head_indices, layer_idx


def add_mlp_scaling(model, layer_idx: int, proportion: float, seed: int=42, scale: float=0.0):
    #Number of hidden layers
    num_layers = model.config.num_hidden_layers

    if not 0 <= layer_idx < num_layers:
        raise InvalidLayerError(
            f"Layer index {layer_idx} is invalid. "
            f"Valid range is 0-{num_layers - 1}."
            )
    
    if not 0 <= proportion <= 1.0:
        raise ValueError(f"Proportion must be between 0 and 1")
    
    # MLP in layer layer_idx
    mlp = model.model.layers[layer_idx].mlp

    num_units = model.config.intermediate_size
    num_scale = int(proportion * num_units)

    # Seed RNG
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Pick 'num_scale' indexes for MLP units at random
    indices = torch.randperm(
        num_units,
        generator=generator
    )[:num_scale]

    def scaling_hook(module, inputs):
        x = inputs[0].clone()
        indices_device = indices.to(x.device)
        x[..., indices_device] *= scale
        return (x,) + inputs[1:]

    handle = mlp.down_proj.register_forward_pre_hook(scaling_hook)

    return handle, indices, layer_idx, proportion


def add_head_noise(model, head_indices: list[int], layer_idx: int, sigma: float, seed: int=42):
    #Number of hidden layers
    num_layers = model.config.num_hidden_layers
    #Number of Query attention heads
    num_heads = model.config.num_attention_heads
    
    if not 0 <= layer_idx < num_layers:
        raise InvalidLayerError(
            f"Layer index {layer_idx} is invalid. "
            f"Valid range is 0-{num_layers - 1}."
            )
    for head_idx in head_indices:
        if not 0 <= head_idx < num_heads:
            raise InvalidHeadError(
                f"Attention head {head_idx} is invalid. "
                f"Valid range is 0-{num_heads - 1}."
                )

    #Projection dimension per attention head
    head_dim = model.model.layers[layer_idx].self_attn.head_dim
    # Load final attention module
    attention_dense = model.model.layers[layer_idx].self_attn.o_proj

    # Set the RNG with the provided seed. Fresh noise sampled on each forward pass.
    generator = torch.Generator(device=attention_dense.weight.device)
    generator.manual_seed(seed)

    def head_noise_hook(module, inputs):
        x = inputs[0].clone()

        for head_idx in head_indices:
            #compute indices corresponding to specific head
            start = head_idx * head_dim
            end = start + head_dim
            #Select indices in the head outputs vector corresponding to chosen head
            selected = x[..., start:end]
            noise = torch.randn_like(selected,generator=generator) * sigma
            x[..., start:end] += noise

        return (x,) + inputs[1:]

    handle = attention_dense.register_forward_pre_hook(head_noise_hook)
    return handle, head_indices, layer_idx, sigma, seed


def add_mlp_noise(model, layer_idx: int, proportion: float, sigma: float, seed: int=42):
    #Number of hidden layers
    num_layers = model.config.num_hidden_layers

    if not 0 <= layer_idx < num_layers:
        raise InvalidLayerError(
            f"Layer index {layer_idx} is invalid. "
            f"Valid range is 0-{num_layers - 1}."
            )
    
    if not 0 <= proportion <= 1.0:
        raise ValueError(f"Proportion must be between 0 and 1")
    
    # MLP in layer layer_idx
    mlp = model.model.layers[layer_idx].mlp

    num_units = model.config.intermediate_size
    num_selected = int(proportion * num_units)

    # Seed RNG for selecting indexes
    generator_idxs = torch.Generator()
    generator_idxs.manual_seed(seed)

    # Pick 'num_selected' indexes for MLP units at random
    indices = torch.randperm(
        num_units,
        generator=generator_idxs
    )[:num_selected]

    # Seed RNG for selecting noise
    generator_noise = torch.Generator(device=mlp.down_proj.weight.device)
    generator_noise.manual_seed(seed+1)

    def mlp_noise_hook(module, inputs):
        x = inputs[0].clone()
        indices_device = indices.to(x.device)
        selected = x[..., indices_device]
        noise = torch.randn_like(selected, generator=generator_noise) * sigma
        x[..., indices_device] += noise
        return (x,) + inputs[1:]

    handle = mlp.down_proj.register_forward_pre_hook(mlp_noise_hook)

    return handle, indices, layer_idx, proportion, sigma, seed