import sys
sys.path.append('../../')
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
try:
    from DISTIL.DISTIL.models.load_models import load_model, get_transform, transform_image
except:
    from DISTIL.models.load_models import load_model, get_transform, transform_image


# Define the custom order for dataset names
dataset_name_order = {"cifar10": 0, "gtsrb": 1, "cifar100": 2, "tiny": 3}

def sort_key(filename):
    """
    Custom sort key: first by predefined dataset order, then alphabetically.
    """
    for dataset_name, rank in dataset_name_order.items():
        if filename.split("_")[0] == dataset_name:
            return rank, filename
    return len(dataset_name_order), filename

class BackbenchDataset(Dataset):
    """
    PyTorch Dataset that loads model data from a base path.
    
    It scans the base_path for model directories, sorts them using sort_key,
    and—for this example—filters to a specific model (as in the original code).
    For each model, it loads the model using load_model() (from load_models.py)
    and samples a small percentage (1%, at least 1) of test images using the standard
    transforms.
    """
    def __init__(self, base_path, target_type="backdoor",model_name=None,dataset_filter=None):
        
        self.base_path = base_path
        # List directories in base_path and sort them
        all_dirs = os.listdir(base_path)
        self.model_dirs = sorted(all_dirs, key=sort_key)
        # adjust filtering as needed:
        if model_name is not None:
            self.model_dirs = [d for d in self.model_dirs if model_name in d]
        else:
            self.model_dirs = [d for d in self.model_dirs]
            
        if dataset_filter:
            self.model_dirs = [d for d in self.model_dirs if d.split('_')[0] == dataset_filter]          
        
        self.target_type = target_type
        
    def __len__(self):
        return len(self.model_dirs)
    
    def __getitem__(self, idx):
        model_name = self.model_dirs[idx]
        # Load the model using the load_model function from load_models.py
        model = load_model(model_name, self.base_path)
        
        bd_test_path = os.path.join(self.base_path, model_name, "bd_test_dataset")
        bd_train_path = os.path.join(self.base_path, model_name, "bd_train_dataset")
        
        # Derive the dataset name from the model name (e.g. "cifar10", "cifar100", etc.)
        dataset_name = model_name.split('_')[0]
        
        # Load the corresponding test dataset.
        if dataset_name == "cifar10":
            test_dataset = datasets.CIFAR10(root="../data", train=False, download=True)
        elif dataset_name == "cifar100":
            test_dataset = datasets.CIFAR100(root="../data", train=False, download=True)
        elif dataset_name == "tiny":
            from DISTIL.DISTIL.data_loader.tiny_dataset import TinyImageNet  # adjust as needed
            test_dataset = TinyImageNet("../data/tiny", 
                                         split='val', download=True)
        elif dataset_name == "gtsrb":
            from DISTIL.DISTIL.data_loader.gtsrb import GTSRB  # adjust as needed
            test_dataset = GTSRB("../data/gtsrb", train=False)
        else:
            raise Exception("Invalid dataset")
        
        # Sample 1% of the test dataset (ensuring at least one sample).
        sample_size = max(1, int(0.01 * len(test_dataset)))
        sample_indices = random.sample(range(len(test_dataset)), sample_size)
        images = []
        labels = []
        for i in sample_indices:
            pil_img, label = test_dataset[i]
            transformed_img = transform_image(pil_img, dataset_name)
            images.append(transformed_img)
            labels.append(label)
        if images:
            images = torch.stack(images)  # Shape: (num_samples, 3, 224, 224)
            labels = torch.tensor(labels)
        
        # Return a dictionary containing metadata and sample data.
        model_data = {
            "model_name": model_name,
            "model_weight": model,
            "images": images,
            "labels": labels,
            "bd_train_path": bd_train_path,
            "bd_test_path": bd_test_path,
            
        }
        return model_data

# Example usage:
if __name__ == "__main__":
    base_path = "" # Backdoorbench dataset path.
    dataset = BackbenchDataset(base_path)
    if len(dataset) > 0:
        model_data = dataset[0]
        print("Loaded model:", model_data["model_name"])
        print("Image sample shape:", model_data["images"].shape)
