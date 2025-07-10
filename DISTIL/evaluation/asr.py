import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

def calculate_class0_percentage(model, folder_path, batch_size=32, transform=None):
    """
    This function loads images from a folder (with subfolders as class labels), runs
    the provided model on the dataset, and returns the percentage of images predicted
    as class 0.
    
    Args:
        model (torch.nn.Module): The pretrained PyTorch model.
        folder_path (str): Path to the dataset folder. Each subfolder corresponds to a class.
        batch_size (int): Batch size to be used by the DataLoader.
    
    Returns:
        float: Percentage of images predicted as class 0.
    """
    
    dataset = torchvision.datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    original_mode = model.training
    model.eval()  
    
    total_images = 0
    class0_count = 0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            
            _, preds = torch.max(outputs, dim=1)
            
            class0_count += (preds == 0).sum().item()
            total_images += images.size(0)
    
    percentage_class0 = (class0_count / total_images) * 100

    if original_mode:
        model.train()
    else:
        model.eval()

    return percentage_class0 