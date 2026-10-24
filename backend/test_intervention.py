import interventions
import pytest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from types import SimpleNamespace

class TinyDownProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self.last_input = None

    def forward(self, x):
        self.last_input = x.clone()
        return self.linear(x)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = TinyDownProj()

class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()
        self.self_attn = TinyAttention()

class TinyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer()])

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInnerModel()
        self.config = SimpleNamespace(
            num_hidden_layers=1,
            intermediate_size=4,
            num_attention_heads=4
        )

class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.o_proj = TinyOutProj()

class TinyOutProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4, bias=False)
        self.last_input = None

    def forward(self, x):
        self.last_input = x.clone()
        return self.linear(x)

def test_head_ablation():
    model = TinyModel()

    handle, head_indices, layer_idx = interventions.add_head_ablation(
                                                model=model, 
                                                head_indices= [0,2], 
                                                layer_idx= 0)
    assert len(head_indices) == 2
    
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
    model.model.layers[0].self_attn.o_proj(x)
    modified = model.model.layers[0].self_attn.o_proj.last_input
    handle.remove()

    assert torch.all(modified[..., [0,1,4,5]] == 0)
    assert torch.equal(modified[..., [2,3,6,7]], x[..., [2,3,6,7]])

def test_mlp_ablation():
    model = TinyModel()

    handle, indices, layer_idx, proportion = interventions.add_mlp_ablation(
                                            model=model, 
                                            layer_idx=0, 
                                            proportion=0.5, 
                                            )
    assert len(indices) == 2
    
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    model.model.layers[0].mlp.down_proj(x)
    modified = model.model.layers[0].mlp.down_proj.last_input
    handle.remove()

    assert torch.all(modified[..., indices] == 0)

    mask = torch.ones(x.shape[-1], dtype=torch.bool)
    mask[indices] = False

    assert torch.equal(modified[..., mask], x[..., mask])

    



