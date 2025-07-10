import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ..models.diffusion_model import generate_image_with_classifier
from ..utils.model_utils import get_images_and_labels_by_label, greedy_class_farthest
from ..utils.model_utils import trigger_evaluation, plot_image_tensor
from ..data_loader.model_loader import get_model_features

def prepare_classifier_for_detection(model_data, is_backdoored=0, model_index=0, interested_class=None, metadata_path=None):
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
    # print(f"Model {model_index} ({model_data['model_id']}) which is {model_sources_names[is_backdoored]} loaded")
    
    images = model_data["images"]
    labels = model_data["labels"]
    classifier = model_data['model_weight'].to(device)
    
    if metadata_path:
        trigger_target = get_model_features(model_data, metadata_path)
    else:
        trigger_target = None
        
    # Filter images and labels if an interested class is provided
    if interested_class is not None:
        images, labels = get_images_and_labels_by_label(images, labels, interested_class)
        
    return classifier, images, labels, trigger_target



def detect_backdoor(model_data, metadata_path=None, guidance_scale=100, num_iterations=2, timestep=50, search_strategy="greedy", model_index=0, add_noise=True, grad_scale_factor=0.142857):
    """
    Detect backdoors in a model using diffusion model-generated triggers.
    
    Args:
        model_data: Model data dictionary
        metadata_path: Path to the metadata CSV file
        guidance_scale: Guidance scale for the diffusion model
        num_iterations: Number of iterations to try different guidance scales
        timestep: Number of timesteps for the diffusion model
        search_strategy: Strategy for finding source-target pairs ('greedy' or 'exhaustive')
        add_noise: Whether to add noise during image transformation
        grad_scale_factor: Scale factor for gradient adjustment
        
    Returns:
        max_score: Maximum backdoor score achieved
        best_trigger: Best trigger found
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier, images, labels, trigger_target = prepare_classifier_for_detection(
        model_data, 
        is_backdoored=(1 if metadata_path else 0), 
        model_index=model_index, 
        metadata_path=metadata_path
    )
    
    # Get all unique class labels
    unique_labels = torch.unique(labels).cpu().numpy()
    num_classes = len(unique_labels)
    
    # Create the list of source-target pairs to evaluate
    if search_strategy == "greedy":
        # Use greedy approach based on classifier weights
        similarity_list = greedy_class_farthest(classifier)
        # print(f"Using greedy search strategy with {len(similarity_list)} target-source pairs")
    elif search_strategy == "exhaustive":
        # Create all possible source-target pairs
        similarity_list = []
        for target_label in unique_labels:
            for source_label in unique_labels:
                if target_label != source_label:  # Exclude self-pairs
                    # Format compatible with greedy_class_farthest output (target, source, dummy_similarity)
                    similarity_list.append((target_label, source_label, 0.0))
        print(f"Using exhaustive search strategy with {len(similarity_list)} target-source pairs")
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}. Use 'greedy' or 'exhaustive'.")
    
    trigger_list = []
    score_list = []
    pair_list = []  # To store which pairs led to which scores
    
    if trigger_target is not None:
        # If we know the target, only evaluate pairs with that target
        filtered_list = [pair for pair in similarity_list if pair[0] == int(trigger_target)]
        if filtered_list:
            similarity_list = filtered_list
            # print(f"Filtered to {len(similarity_list)} pairs with target class {trigger_target}")
        else:
            print(f"Warning: No pairs found with target class {trigger_target}. Using all pairs.")
            
    for pair_index, (target_label, source_label, _) in enumerate(similarity_list):
        # print(f"Evaluating pair {pair_index+1}/{len(similarity_list)}: Target={target_label}, Source={source_label}")
        
        target_image, _ = get_images_and_labels_by_label(images, labels, target_label)
        if len(target_image) == 0:
            print(f"Warning: No images found for target class {target_label}. Skipping pair.")
            continue
        
        target_image = target_image[0:1]  # Select first image
        
        source_images, _ = get_images_and_labels_by_label(images, labels, source_label)
        if len(source_images) == 0:
            print(f"Warning: No images found for source class {source_label}. Skipping pair.")
            continue
            
        source_images = source_images[0:1]        
        
        current_guidance_scale = guidance_scale
        for iteration in range(num_iterations):
            # print(f"Guidance scale: {current_guidance_scale}")
            
            trigger, trigger_confidence = generate_image_with_classifier(
                classifier, 
                timestep,
                current_guidance_scale,
                target_label,
                target_image, 
                source_label,  # Note: Changed from source_labels to source_label
                add_noise=add_noise,  # Pass add_noise
                grad_scale_factor=grad_scale_factor  # Pass grad_scale_factor
            )
            
            if trigger_confidence > 0.95 or iteration == num_iterations - 1:
                # print(f"Target: {target_label}, Source: {source_label}")
                
                score = trigger_evaluation(classifier, trigger.detach().cpu(), images, target_label)
                score_list.append(score)
                trigger_list.append(trigger)
                pair_list.append((target_label, source_label))
                
                # Visualize the trigger
                # plot_image_tensor(trigger.cpu())
                # print(f"Trigger confidence: {trigger_confidence:.4f}, Score: {score:.4f}")
                break
            else:
                current_guidance_scale *= 1.5
                
    # Return the maximum score and corresponding trigger
    if score_list:
        max_score_index = np.argmax(score_list)
        max_score = score_list[max_score_index]
        best_trigger = trigger_list[max_score_index]
        best_pair = pair_list[max_score_index]
        # print(f"Best pair found: Target={best_pair[0]}, Source={best_pair[1]} with score {max_score:.4f}")
        return max_score, best_trigger
    else:
        return 0.0, None


def evaluate_backdoor_detection(clean_models, backdoor_models, metadata_path, output_file=None):
    """
    Evaluate backdoor detection on multiple models.
    
    Args:
        clean_models: List of clean model data
        backdoor_models: List of backdoored model data
        metadata_path: Path to the metadata CSV file
        output_file: Path to save evaluation results
        
    Returns:
        results: Dictionary of detection results
    """
    results = {}
    
    # Evaluate on clean models
    for i in range(len(clean_models)):
        print(f"Evaluating clean model {i}")
        max_score, _ = detect_backdoor(clean_models[i])
        results[f"clean_{i}"] = max_score
        
        # Write to output file
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"model {i} 0 {max_score} \n\n")
                
    # Evaluate on backdoored models
    for i in range(len(backdoor_models)):
        print(f"Evaluating backdoored model {i}")
        max_score, _ = detect_backdoor(backdoor_models[i], metadata_path)
        results[f"backdoored_{i}"] = max_score
        
        # Write to output file
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"model {i} 1 {max_score} \n\n")
    
    return results 