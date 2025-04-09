import torch
from safetensors.torch import load_file, save_file

def save_checkpoint(model, filename):
    state_dict = model.state_dict()
    save_file(state_dict, filename)
    print(f"Checkpoint saved to: {filename}")

def load_checkpoint(model, filename):
    state_dict = load_file(filename)
    assert isinstance(model, torch.nn.Module), "Please provide a valid torch model."
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {filename}")
