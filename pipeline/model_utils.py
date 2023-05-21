from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, path: Path, model_name):
    """
    Function to save the trained model to disk.
    """
    torch.save(model.state_dict(), path / f'{model_name}.pt')


def load_model(model: nn.Module, path: Path, model_name):
    """
    Function to load the model.
    """
    model.load_state_dict(torch.load(path / f'{model_name}.pt'))
    return model
