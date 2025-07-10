"""
Image transformation utilities for object detection models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# Default image transformation pipeline
DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_DEFAULT_IMAGE_TRANSFORM():
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

def get_dataset_normalization(dataset_name):
    """
    Get the normalization parameters for a given dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        dataset_normalization: Transforms.Normalize for the dataset
    """
    import torchvision.transforms as transforms
    
    if dataset_name == "cifar10":
        # from wanet
        dataset_normalization = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif dataset_name == 'cifar100':
        dataset_normalization = transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    elif dataset_name == "mnist":
        dataset_normalization = transforms.Normalize([0.5], [0.5])
    elif dataset_name == 'tiny':
        dataset_normalization = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        raise Exception(f"Invalid Dataset: {dataset_name}")
    
    return dataset_normalization

def get_transform(dataset_name, input_height, input_width, train=True, random_crop_padding=4):
    """
    Get the transform for a given dataset.
    
    Args:
        dataset_name: Name of the dataset
        input_height: Height of the input image
        input_width: Width of the input image
        train: Whether this is for training or testing
        random_crop_padding: Padding for random crop
        
    Returns:
        transforms: Composed transforms for the dataset
    """
    import torchvision.transforms as transforms
    
    transforms_list = []
    transforms_list.append(transforms.Resize((224, 224)))
    
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
        # transforms_list.append(transforms.RandomRotation(10))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(dataset_name))
    
    return transforms.Compose(transforms_list)

def transform_image(image, dataset_name, input_height=224, input_width=224):
    """
    Apply transformation to an image based on dataset.
    
    Args:
        image: PIL image to transform
        dataset_name: Name of the dataset
        input_height: Height of the input image
        input_width: Width of the input image
        
    Returns:
        tensor_image: Transformed image as tensor
    """
    transform = get_transform(dataset_name, input_height, input_width, train=False, random_crop_padding=4)
    tensor_image = transform(image)
    return tensor_image

def transform_image_tensor(image):
    """
    Convert an input image to a tensor of shape [1, C, 224, 224].
    If the input is a numpy.ndarray, convert it to a PIL Image first.
    
    Args:
        image: A numpy array, PIL Image, or tensor of shape [C, H, W].
        
    Returns:
        A tensor of shape [1, C, 224, 224].
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    if isinstance(image, torch.Tensor) and image.ndim == 3:
        image = image.unsqueeze(0)
    return F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False) 