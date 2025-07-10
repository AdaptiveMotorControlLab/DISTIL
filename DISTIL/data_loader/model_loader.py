import os
import json
import warnings
import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image

try:
    from DISTIL.DISTIL.utils.transforms import transform_image_tensor,get_DEFAULT_IMAGE_TRANSFORM
except:
    from DISTIL.utils.transforms import transform_image_tensor,get_DEFAULT_IMAGE_TRANSFORM

class TrojAIModelBatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_models=None,poison=True):
        """
        Args:
            root_dir (string): Root directory containing id-0000000x folders
            transform (callable, optional): Optional transform to be applied
            max_models (int, optional): Limit number of models loaded
        """
        self.root_dir = root_dir
        self.transform = transform
        self.model_records = []  # Stores metadata per model

        # First level: id-0000000x folders
        for x_folder in sorted(os.listdir(root_dir)):
            if not x_folder.startswith('id-') or not os.path.isdir(os.path.join(root_dir, x_folder)):
                continue

            # Second level: id-00000000 folders
            x_path = os.path.join(root_dir, x_folder)
            for model_folder in sorted(os.listdir(x_path)):
                if not model_folder.startswith('id-') or not os.path.isdir(os.path.join(x_path, model_folder)):
                    continue

                model_path = os.path.join(x_path, model_folder)
                
                try:
                    torch.load(os.path.join(model_path, 'model.pt'),weights_only=False)
                except:
                    print(f"Warning: Could not load model for {model_path}")
                    continue

                # Read ground truth
                gt_file = os.path.join(model_path, 'ground_truth.csv')
                try:
                    ground_truth = pd.read_csv(gt_file, header=None).values[0][0]
                    if ground_truth!=poison:
                        continue
                except:
                    print(f"Warning: Could not read ground truth for {model_path}")
                    continue

                # Get all example images
                if os.path.exists(os.path.join(model_path, 'clean-example-data')):
                    example_dir = os.path.join(model_path, 'clean-example-data')
                elif os.path.exists(os.path.join(model_path, 'example_data')):
                    example_dir = os.path.join(model_path, 'example_data')
                else:
                    print(f"Warning: No clean-example-data in {model_path}")
                    continue

                model_images = []
                for img_name in sorted(os.listdir(example_dir)):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            class_label = int(img_name.split('_')[1])  # Extract class
                            img_path = os.path.join(example_dir, img_name)
                            model_images.append((img_path, class_label))
                        except (IndexError, ValueError):
                            continue

                if model_images:
                    self.model_records.append({
                        'model_path': model_path,
                        'ground_truth': ground_truth,
                        'available_images': model_images,
                        'model_id': os.path.basename(model_path),
                        'parent_id': os.path.basename(os.path.dirname(model_path))
                    })

                if max_models and len(self.model_records) >= max_models:
                    break
            if max_models and len(self.model_records) >= max_models:
                break

    def __len__(self):
        return len(self.model_records)  # Dataset length is number of models

    def __getitem__(self, idx):
        model_info = self.model_records[idx]
        images = []
        labels = []

        for img_path, class_label in model_info['available_images']:
            # Load image
            image = cv2.imread(img_path)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            images.append(image)
            labels.append(class_label)

        return {
            'images': torch.stack(images),  # Returns all images as a batch
            'labels': torch.tensor(labels),  # Corresponding labels
            'ground_truth': model_info['ground_truth'],
            'model_id': model_info['model_id'],
            'parent_id': model_info['parent_id'],
            'model_path': model_info['model_path'],
            'model_weight': torch.load(os.path.join(model_info['model_path'], 'model.pt'),weights_only=False),
        }


def load_models(base_path,limit_model=None):
    """
    Load models from the given path and separate them into clean and backdoored models.
    
    Args:
        base_path: Base directory containing model folders
        
    Returns:
        clean_models: List of clean model data
        backdoor_models: List of backdoored model data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
    
    
    clean_models=TrojAIModelBatchDataset(root_dir=base_path,transform=get_DEFAULT_IMAGE_TRANSFORM(),poison=False,max_models=limit_model)
    backdoor_models=TrojAIModelBatchDataset(root_dir=base_path,transform=get_DEFAULT_IMAGE_TRANSFORM(),poison=True,max_models=limit_model)

    
    return clean_models, backdoor_models


def get_model_features(model_data, metadata_path):
    """
    Get features of a model from the metadata CSV file.
    
    Args:
        model_data: Dictionary containing model information
        metadata_path: Path to the metadata CSV file
        
    Returns:
        trigger_target_class: The target class of the backdoor
    """
    df = pd.read_csv(metadata_path, index_col=0)
    
    model_id =  model_data["model_id"]
    
    if model_id in df.index:
        row = df.loc[model_id]
        print(f"{model_data['model_id']} triggered_classes: {row['triggered_classes']} trigger_target_class: {row['trigger_target_class']}")
        return row['trigger_target_class']
    else:
        raise ValueError(f"Model id '{model_id}' not found in the CSV file.")


def load_backdoored_models(base_path, model_filter=None, dataset_filter=None, limit=None):
    """
    Load backdoored models from a specified path with optional filtering.
    
    Args:
        base_path: Path to directory containing backdoored models
        model_filter: Optional filter for model architectures
        dataset_filter: Optional filter for datasets
        limit: Optional limit on number of models to load
    
    Returns:
        backdoor_models: List of dictionaries containing model data
    """
    import os
    import torch
    from torchvision.models import vgg19_bn, convnext_tiny, vit_b_16
    from torchvision.transforms import Resize
    import random
    from DISTIL.DISTIL.models.preact_resnet import PreActResNet18
    
    # Map dataset names to number of classes
    dataset_to_classes = {
        "cifar10": 10,
        "gtsrb": 43,
        "celeba": 8,
        "cifar100": 100,
        "tiny": 200,
        "imagenet": 1000
    }
    
    # Sort files by dataset
    files = os.listdir(base_path)
    dataset_name_order = {"cifar10": 0, "gtsrb": 1, "cifar100": 2, "tiny": 3}
    
    def sort_key(filename):
        dataset = filename.split("_")[0]
        rank = dataset_name_order.get(dataset, len(dataset_name_order))
        return (rank, filename)
    
    sorted_files = sorted(files, key=sort_key)
    
    # Apply filters if provided
    if model_filter:
        sorted_files = [f for f in sorted_files if model_filter in f.split('_')[1]]
    
    if dataset_filter:
        sorted_files = [f for f in sorted_files if f.split('_')[0] == dataset_filter]
    
    # Apply limit if provided
    if limit is not None:
        sorted_files = sorted_files[:limit]
    
    backdoor_models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name in sorted_files:
        dataset_name = model_name.split('_')[0]
        model_arch = model_name.split('_')[1]
        
        # Skip if "blind" is in the model name (blind models)
        if "blind" in model_name.split('_'):
            continue
        
        print(f"Loading model: {model_name}, dataset: {dataset_name}, architecture: {model_arch}")
        
        file_path = os.path.join(base_path, model_name, "attack_result.pt")
        
        # Skip if the file doesn't exist
        if not os.path.exists(file_path):
            print(f"Skipping {model_name}, file not found: {file_path}")
            continue
        
        # Determine number of classes for the dataset
        if dataset_name not in dataset_to_classes:
            print(f"Skipping unknown dataset: {dataset_name}")
            continue
            
        num_classes = dataset_to_classes[dataset_name]
        
        # Create model based on architecture
        model = None
        if model_arch == "preactresnet18":
            model = PreActResNet18(num_classes=num_classes)
        elif model_arch == "vgg19":
            model = vgg19_bn(num_classes=num_classes)
        elif model_arch == "convnext":
            model = convnext_tiny(num_classes=num_classes)
        elif model_arch == "vit":
            model = vit_b_16(pretrained=True)
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, out_features=num_classes, bias=True)
            
            # Wrap the model in a Sequential with a Resize layer
            model = torch.nn.Sequential(
                Resize((224, 224)),  # Resize input images to 224x224
                model,
            )
        else:
            print(f"Unsupported model architecture: {model_arch}")
            continue
        
        # Load model weights
        try:
            loaded_file = torch.load(file_path, map_location=device)
            model.load_state_dict(loaded_file["model"])
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        # Sample test images for evaluation
        from DISTIL.DISTIL.utils.transforms import transform_image
        from torchvision import datasets
        from DISTIL.DISTIL.data_loader.tiny_imagenet import TinyImageNet
        from DISTIL.DISTIL.data_loader.gtsrb import GTSRB
        
        # Load test dataset based on dataset name
        test_dataset = None
        if dataset_name == "cifar10":
            test_dataset = datasets.CIFAR10(root="../data", train=False, download=True)
        elif dataset_name == "cifar100":
            test_dataset = datasets.CIFAR100(root="../data", train=False, download=True)
        elif dataset_name == "tiny":
            test_dataset = TinyImageNet(root="../data/tiny-imagenet", split='val', download=True)
        elif dataset_name == "gtsrb":
            test_dataset = GTSRB(data_root="../data/gtsrb", train=False)
        else:
            print(f"Dataset {dataset_name} not supported for test sample extraction")
            continue
        
        # Determine sample size (1% of test set)
        sample_size = max(1, int(0.01 * len(test_dataset)))
        
        # Randomly select indices from test dataset
        sample_indices = random.sample(range(len(test_dataset)), sample_size)
        
        images = []
        labels = []
        
        for idx in sample_indices:
            try:
                # Get image and label
                pil_img, label = test_dataset[idx]
                # Transform the image
                transformed_img = transform_image(pil_img, dataset_name)
                images.append(transformed_img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing test sample {idx}: {e}")
                continue
        
        # Convert lists to tensors
        if images:
            images = torch.stack(images)  # Shape: (num_images, 3, 224, 224)
            labels = torch.tensor(labels)  # Shape: (num_images,)
        else:
            print(f"No valid test samples for {model_name}")
            continue
        
        # Store model data
        model_data = {
            "model_name": model_name,
            "model": model,
            "images": images,
            "labels": labels
        }
        
        backdoor_models.append(model_data)
    
    print(f"\nLoaded {len(backdoor_models)} backdoored models")
    return backdoor_models 