import torch
import warnings
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, vit_b_16, vgg19_bn
from torchvision.transforms import Resize
import os
import numpy as np

from .preact_resnet import PreActResNet18
from ..data_loader.tiny_imagenet import TinyImageNet
from ..data_loader.gtsrb import GTSRB
from ..utils.transforms import transform_image_tensor


def get_num_classes(dataset_name: str) -> int:
    if dataset_name in ["mnist", "cifar10"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tiny':
        num_classes = 200
    elif dataset_name == 'imagenet':
        num_classes = 1000
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_dataset_normalization(dataset_name):
    if dataset_name == "cifar10":
        dataset_normalization = (transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset_name == 'cifar100':
        dataset_normalization = (transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]))
    elif dataset_name == "mnist":
        dataset_normalization = (transforms.Normalize([0.5], [0.5]))
    elif dataset_name == 'tiny':
        dataset_normalization = (transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]))
    elif dataset_name == "gtsrb" or dataset_name == "celeba":
        dataset_normalization = transforms.Normalize([0, 0, 0], [1, 1, 1])
    elif dataset_name == 'imagenet':
        dataset_normalization = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        raise Exception("Invalid Dataset")
    return dataset_normalization


def get_transform(dataset_name, input_height, input_width, train=True, random_crop_padding=4):
    transforms_list = []
    transforms_list.append(transforms.Resize((224, 224)))
    if train:
        transforms_list.append(transforms.RandomCrop((input_height, input_width), padding=random_crop_padding))
        if dataset_name == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(dataset_name))
    return transforms.Compose(transforms_list)


def transform_image(image, dataset_name, input_height=224, input_width=224):
    transform = get_transform(dataset_name, input_height, input_width, train=False, random_crop_padding=4)
    tensor_image = transform(image)
    return tensor_image


def load_model(model_name, base_path):
    dataset_name = model_name.split('_')[0]
    model_arch = model_name.split('_')[1]

    num_classes = get_num_classes(dataset_name)
    if model_arch == "preactresnet18":
        model = PreActResNet18(num_classes=num_classes)
    elif model_arch == "vgg19":
        model = vgg19_bn(num_classes=num_classes)
    elif model_arch == "convnext":
        model = convnext_tiny(num_classes=num_classes)
    elif model_arch == "vit":
        model = vit_b_16(pretrained=True)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, out_features=num_classes, bias=True)

        model = torch.nn.Sequential(
            Resize((224, 224)),
            model,
        )
    else:
        raise Exception("Invalid Model Architecture")

    file_path = os.path.join(base_path, model_name, "attack_result.pt")
    # with torch.serialization.safe_globals([np.core.multiarray.scalar]):
    #     loaded_file = torch.load(file_path, weights_only=True)
    loaded_file = torch.load(file_path, weights_only=False)
    model.load_state_dict(loaded_file["model"])

    return model