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
        raise ValueError(f"proportion must be between 0 and 1")
    
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