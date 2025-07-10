import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../../')
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

# Import utility functions from your project
from DISTIL.DISTIL.utils.model_utils import (
    greedy_class_farthest,
    trigger_evaluation,
    plot_image_tensor,)

from DISTIL.DISTIL.evaluation.asr import calculate_class0_percentage
from DISTIL.DISTIL.evaluation.test_accuracy import test_model
from DISTIL.DISTIL.models.load_models import get_transform
from DISTIL.DISTIL.models.diffusion_model_upsample import run_diffusion_with_trigger
from DISTIL.DISTIL.utils.model_utils import get_images_and_labels_by_label, greedy_class_farthest
from DISTIL.DISTIL.data_loader import model_loader,backbench
from DISTIL.DISTIL.detection.backdoor_target_detector import BackdoorTargetDetector


# Global output file for logging results
OUTPUT_FILE = "./backdoored_models_mitigation_result"


def prepare_classifier_for_mitigation(model_data, is_backdoored=0, model_index=0, interested_class=None, metadata_path=None):
    """
    Prepare a classifier model for backdoor detection.
    
    Args:
        model_data: Model data dictionary
        is_backdoored: Whether the model is known to be backdoored (0=clean, 1=backdoor)
        model_index: Index of the model in the list
        interested_class: Specific class of interest for backdoor detection
        metadata_path: Path to the metadata CSV file
        
    Returns:
        classifier: Prepared classifier model
        images: Tensor of relevant images
        labels: Tensor of relevant labels
        trigger_target: Target class of the backdoor
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_sources_names = ["clean", "backdoored"]
    print(f"Model {model_index} loaded")
    
    images = model_data["images"]
    labels = model_data["labels"]
    classifier = model_data['model_weight'].to(device)
    
    if metadata_path:
        trigger_target = model_loader.get_model_features(model_data, metadata_path)
    else:
        trigger_target = None
        
    # Filter images and labels if an interested class is provided
    if interested_class is not None:
        images, labels = get_images_and_labels_by_label(images, labels, interested_class)
        
    return classifier, images, labels, trigger_target


def generate_augmented_dataset(classifier, images, labels, predicted_target_label):
    """
    Generate an augmented dataset by combining original images with generated triggers.
    
    For each image (and its label) in the input, a trigger is generated using the diffusion 
    upsampling process. The augmented dataset will consist of both the original image and the 
    generated trigger (with the same label).
    
    Args:
        classifier (torch.nn.Module): The classifier used to generate the trigger.
        images (iterable): Original images.
        labels (iterable): Corresponding labels.
        
    Returns:
        augmented_images (list): Collection of images including originals and triggers.
        augmented_labels (list): Corresponding labels.
    """
    similarity_list = greedy_class_farthest(classifier)
    print(f"Found {len(similarity_list)} similarity pairs: {similarity_list}")
    
    augmented_images = []
    augmented_labels = []
    
    # Loop through each image and label.
    for idx, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images), desc="Augmenting dataset"):
        # For each image, we generate a trigger using the first similarity pair.
        target_label = predicted_target_label
        guid_scale = 150  # initial guidance scale
        trigger = None
        trigger_confidence = 0.0
        # Generate trigger for a couple of iterations.
        for iteration in range(2):
            if idx < 4: # Just print output for first 4 to avoid long output in the notebook.
                print("guid_scale:", guid_scale)

            trigger, trigger_confidence = run_diffusion_with_trigger(
                image.unsqueeze(0),
                classifier,
                target_label=target_label,
                source_label=label,
                guidance_scale=guid_scale,
                upsample_temp=0.997
            )
            if trigger_confidence > 0.95 or iteration == 1:
                score_ = trigger_evaluation(classifier, trigger.detach().cpu(), images, target_label)
                if idx < 4: # Just print output for first 4 to avoid long output in the notebook.
                    print(f"Trigger generated with confidence {trigger_confidence:.4f} for label {label}")
                    print("Evaluation score:", score_)
                    # Optionally, display the image.
                    "Original image:"
                    plot_image_tensor(image)
                    # Optionally plot the trigger.
                    "Image with generated trigger:"
                    plot_image_tensor(trigger.cuda())
                break
            else:
                guid_scale *= 1.5
        
        # Add the original image and its trigger to the augmented dataset.
        augmented_images.append(image)
        augmented_labels.append(label)
        augmented_images.append(trigger.squeeze(0))
        augmented_labels.append(label)
        
        
    return augmented_images, augmented_labels


class AugmentedDataset(Dataset):
    """
    Dataset for images augmented with generated triggers.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].cuda()
        lbl = self.labels[idx].cuda()
        return img, lbl


def fine_tune_classifier(model, dataloader, num_epochs=10, learning_rate=1e-5, device="cpu"):
    """
    Finetune the given model on the provided augmented dataset.
    
    Args:
        model (torch.nn.Module): The classifier to be finetuned.
        dataloader (DataLoader): Dataloader for the augmented dataset.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for finetuning.
        device (str or torch.device): Device for training.
        
    Returns:
        torch.nn.Module: The finetuned model.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model


def evaluate_mitigated_model(model, test_dataset, transform, device,bd_test_path):
    """
    Evaluate the finetuned model on the test dataset.
    
    Args:
        model (torch.nn.Module): The finetuned model.
        test_dataset: Test dataset.
        transform: Transformation to apply to the test dataset.
        device: Device to perform evaluation.
        
    Returns:
        tuple: (test accuracy, backdoor test attack success rate)
    """
    test_acc = test_model(model, test_dataset, batch_size=32, transform=transform, device=device)
    bd_test_asr = calculate_class0_percentage(model, bd_test_path, batch_size=32, transform=transform)
    return test_acc, bd_test_asr


def mitigate_backdoor_model(model_data, metadata_path=None, guidance_scale=100, num_iterations=2, timestep=50, search_strategy="greedy", model_index=0, add_noise=True, grad_scale_factor=0.142857):
    """
    Mitigate a single model given its index.
    
    Args:
        model_data: Dictionary containing model and data information.
        metadata_path: Path to metadata file.
        guidance_scale: Scale for diffusion guidance.
        num_iterations: Number of iterations for trigger generation.
        timestep: Timestep for diffusion.
        search_strategy: Strategy for class pair search.
        model_index: Index of the model to mitigate.
        add_noise: Whether to add noise in diffusion process.
        grad_scale_factor: Scale factor for gradient calculations.
        
    Returns:
        Tuple of (model_name, final_test_acc, final_bd_test_asr)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare classifier and data.
    classifier, images, labels, trigger_target = prepare_classifier_for_mitigation(
        model_data, 
        is_backdoored=(1 if metadata_path else 0), 
        model_index=model_index, 
        metadata_path=metadata_path
    )


    predicted_target_label, _, _ = BackdoorTargetDetector(output_file=OUTPUT_FILE).detect_target_class(model_data)
    
    # Generate augmented dataset with generated triggers.
    augmented_images, augmented_labels = generate_augmented_dataset(classifier, images, labels, predicted_target_label)
    
    # Create dataset and dataloader for finetuning.
    aug_dataset = AugmentedDataset(augmented_images, augmented_labels)
    dataloader = DataLoader(aug_dataset, batch_size=32, shuffle=True)
    
    # Determine dataset details from model_name.
    dataset_name = model_data['model_name'].split('_')[0]
    if dataset_name == "cifar10":
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    elif dataset_name == "cifar100":
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True)
    elif dataset_name == "tiny":
        from DISTIL.DISTIL.data_loader.tiny_imagenet import TinyImageNet  # adjust as needed
        Tiny_dataset_address = "../data/tiny" # set this based on dataset path on your system
        test_dataset = TinyImageNet(Tiny_dataset_address,
                                    split='val', download=True)
    elif dataset_name == "gtsrb":
        from DISTIL.DISTIL.data_loader.gtsrb import GTSRB  # adjust as needed
        GTSRB_dataset_address = "../data/gtsrb" # set this based on dataset path on your system
        test_dataset = GTSRB(GTSRB_dataset_address, train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_transform = get_transform(dataset_name, input_height=224, input_width=224, train=False)
    
    # Optionally, evaluate before mitigation.
    print("Evaluating before mitigation:")
    test_acc_before = test_model(classifier, test_dataset, batch_size=32, transform=test_transform, device=device)
    bd_test_asr_before = calculate_class0_percentage(classifier, model_data['bd_test_path'], batch_size=32, transform=test_transform)
    print(f"Before mitigation: test_acc: {test_acc_before*100:.2f}%, bd_test_asr: {bd_test_asr_before:.2f}%")
    
    # Finetune (mitigate) the classifier.
    mitigated_model = copy.deepcopy(classifier).to(device)
    mitigated_model = fine_tune_classifier(mitigated_model, dataloader, num_epochs=10, learning_rate=1e-5, device=device)
    
    # Evaluate the mitigated model.
    test_acc_after, bd_test_asr_after = evaluate_mitigated_model(mitigated_model, test_dataset, test_transform, device,bd_test_path=model_data['bd_test_path'])
    print(f"After mitigation: test_acc: {test_acc_after*100:.2f}%, bd_test_asr: {bd_test_asr_after:.2f}%")
    return model_data['model_name'], test_acc_after, bd_test_asr_after


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assume backdoor_models is defined/loaded elsewhere in your project.
    # For mitigation, we iterate over the backdoored models.
    # Here we assume backdoor_models is a list or similar.
    # backdoor_models = backbench.BackbenchDataset(
    #     root_dir='',)
    
    base_path = "" # Backdoorbench dataset path
    backdoor_models = backbench.BackbenchDataset(base_path,model_name='cifar10_vgg19_bn_ssba_0_1')
        
    for model_index in range(len(backdoor_models)):
        is_backdoored = 1
        model_name, test_acc, bd_test_asr = mitigate_backdoor_model(model_data=backdoor_models[model_index],model_index=model_index)
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"{model_name} -> bd_test_asr: {bd_test_asr:.2f}%, test_acc: {test_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()


