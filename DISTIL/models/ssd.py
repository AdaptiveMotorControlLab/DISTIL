"""
SSD (Single Shot MultiBox Detector) model implementation with custom loading mechanisms.
"""

import sys
import types
import pickle
import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16

class SSD(nn.Module):
    """
    SSD model wrapper for custom pickling logic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Instantiate the SSD model using default weights (set to None)
        self.model = ssd300_vgg16(weights=None)

    def forward(self, x):
        # Ensure input and model are on the same device by checking first parameter
        model_device = next(self.model.parameters()).device
        if model_device != x.device:
            self.model = self.model.to(x.device)
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict)

# Register dummy module for pickling references to "models.SSD"
dummy_models = types.ModuleType("models")
dummy_models.SSD = SSD
sys.modules["models"] = dummy_models

# Custom unpickler to resolve models.SSD correctly during unpickling
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models" and hasattr(dummy_models, name):
            return getattr(dummy_models, name)
        return super().find_class(module, name)

def my_load(f, map_location=None):
    return CustomUnpickler(f).load()

# Create a dummy pickle module with our custom unpickler and load function
dummy_pickle = type("dummy_pickle", (), {"Unpickler": CustomUnpickler, "load": my_load})

def load_ssd_checkpoint(ckpt_path, device=None):
    """
    Load an SSD checkpoint using custom pickling logic.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to load the model on.
    
    Returns:
        An SSD model instance if successful; otherwise, None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, pickle_module=dummy_pickle)
    except (AttributeError, RuntimeError) as e:
        print(f" [SKIP] Could not unpickle '{ckpt_path}' as SSD: {e}")
        return None

    # Skip if the checkpoint string indicates a reference to FasterRCNN instead of SSD.
    if "FasterRCNN" in str(checkpoint):
        print(f" [SKIP] '{ckpt_path}' references FasterRCNN, not SSD.")
        return None

    model = SSD()
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        print(f" [SKIP] '{ckpt_path}' unsupported checkpoint type: {type(checkpoint)}")
        return None

    try:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f" [SKIP] '{ckpt_path}' is not an SSD or has a mismatch: {e}")
        return None 