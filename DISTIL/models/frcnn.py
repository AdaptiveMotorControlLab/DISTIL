"""
Faster R-CNN model implementation with custom loading mechanisms.
"""

import sys
import types
import pickle
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNN(nn.Module):
    """
    FasterRCNN model wrapper with custom pickling support.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Instantiate the FasterRCNN model. Use weights=None for custom loading.
        self.model = fasterrcnn_resnet50_fpn(weights=None)

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict)

# Dummy implementations for submodules that might be referenced during unpickling.
class RegionProposalNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class RoIHeads(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None

# Register dummy models within a dummy module for pickling references to "models.FasterRCNN",
# as well as its ancillary components.
dummy_models = types.ModuleType("models")
dummy_models.FasterRCNN = FasterRCNN
dummy_models.RegionProposalNetwork = RegionProposalNetwork
dummy_models.RoIHeads = RoIHeads
sys.modules["models"] = dummy_models

# Custom unpickler for FasterRCNN using the dummy module.
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models" and hasattr(dummy_models, name):
            return getattr(dummy_models, name)
        return super().find_class(module, name)

def my_load(f, map_location=None):
    return CustomUnpickler(f).load()

dummy_pickle = type("dummy_pickle", (), {"Unpickler": CustomUnpickler, "load": my_load})

def remap_fastrcnn_state_dict(state_dict):
    """
    Remap state dictionary keys for FasterRCNN checkpoints with a slightly different key structure.
    
    Args:
        state_dict (dict): The original state dictionary.
    
    Returns:
        dict: The remapped state dictionary.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.fpn.inner_blocks."):
            parts = k.split('.')
            if len(parts) == 5:
                new_k = ".".join(parts[:3] + [parts[3], "0"] + parts[4:])
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        elif k.startswith("backbone.fpn.layer_blocks."):
            parts = k.split('.')
            if len(parts) == 5:
                new_k = ".".join(parts[:3] + [parts[3], "0"] + parts[4:])
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        elif k.startswith("rpn.head.conv") and ("." not in k[len("rpn.head.conv"):]):
            new_k = "rpn.head.conv.0.0" + k[len("rpn.head.conv"):]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_frcnn_checkpoint(ckpt_path, device=None):
    """
    Load a FasterRCNN checkpoint using custom pickling logic.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to load the model on.
    
    Returns:
        A FasterRCNN model instance if successful; otherwise, None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, pickle_module=dummy_pickle)
    except (AttributeError, RuntimeError) as e:
        print(f" [SKIP] Could not unpickle '{ckpt_path}' as FasterRCNN: {e}")
        return None

    # Optionally: skip if the checkpoint indicates it is for the other architecture.
    if "SSD" in str(checkpoint):
        print(f" [SKIP] '{ckpt_path}' references SSD, not FasterRCNN.")
        return None

    model = FasterRCNN()
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        print(f" [SKIP] '{ckpt_path}' unsupported checkpoint type: {type(checkpoint)}")
        return None

    try:
        state_dict = remap_fastrcnn_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f" [SKIP] '{ckpt_path}' is not a FasterRCNN or has a mismatch: {e}")
        return None

def fix_fasterrcnn_classifier(model, num_classes=91):
    """
    Ensure that the FasterRCNN classifier head (box predictor) has the expected number of classes.
    If not, replace it with a new FastRCNNPredictor.
    
    Args:
        model (nn.Module): The FasterRCNN model.
        num_classes (int): The expected number of classes.
        
    Returns:
        nn.Module: The updated model.
    """
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):
        predictor = model.model.roi_heads.box_predictor
        if predictor.cls_score.weight.shape[0] < num_classes:
            in_features = predictor.cls_score.in_features
            model.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            print(f"Replaced FasterRCNN classifier head with {num_classes} classes.")
    return model 