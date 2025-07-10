import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt

from DISTIL.DISTIL.models.diffusion_model import generate_image_with_classifier
from DISTIL.DISTIL.utils.model_utils import get_classifier_layer, trigger_evaluation, get_images_and_labels_by_label, greedy_class_farthest, plot_image_tensor

class BackdoorTargetDetector:
    """
    Class to detect the target class of a backdoored model using diffusion-based trigger generation.
    """
    
    def __init__(self, output_file=None):
        """
        Initialize the detector.
        
        Args:
            output_file: Path to output file to save results (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_file = output_file
        
    def detect_target_class(self, model_data):
        """
        Detect the target class of a backdoored model.
        
        Args:
            model_data: Dictionary containing model information
            
        Returns:
            predicted_target_class: The predicted target class
            score_list: List of scores for each class
            trigger: The generated trigger for the predicted target class
        """
        model_name = model_data["model_name"]
        model = model_data["model_weight"].to(self.device)
        images = model_data["images"]
        labels = model_data["labels"]
        
        print(f"Model name: {model_name}")
        
        similarity_list = greedy_class_farthest(model)
        
            
        print(f"Examining {len(similarity_list)} potential target classes")
        
        score_list = []
        trigger_list = []
        
        for trigger_index, similarity_pair in enumerate(similarity_list):
            target_label, source_labels, _ = similarity_pair
            
            target_image, _ = get_images_and_labels_by_label(images, labels, target_label)
            target_image = target_image[0:1] if len(target_image) > 0 else torch.randn(1, 3, 224, 224).to(self.device)
            
            source_images, _ = get_images_and_labels_by_label(images, labels, source_labels)
            source_images = source_images[0:1] if len(source_images) > 0 else torch.randn(1, 3, 224, 224).to(self.device)
            
            # Start with reasonable guidance scale
            guid_scale = 150
            success = False
            
            for iteration in range(2):
                if iteration > 0:
                    print(f"Increasing guid_scale")
                # print(f"Trying guid_scale: {guid_scale} for target class {target_label}")
                
                trigger, trigger_confidence = generate_image_with_classifier(
                    model, 50, guid_scale, target_label, target_image, source_labels, source_images
                )
                
                if trigger_confidence > 0.95 or iteration == 1:
                    score = trigger_evaluation(model, trigger.detach().cpu(), images, target_label)
                    score_list.append(score)
                    trigger_list.append(trigger.detach().cpu())
                    if target_label < 4: # Just print output for first 4 classes to avoid long output in the notebook.
                        print(f"Target class: {target_label}, Source class: {source_labels}")
                        plot_image_tensor(trigger.cuda())
                        print(f"Trigger confidence: {trigger_confidence}")
                    success = True
                    break
                else:
                    guid_scale *= 1.5
                    
            if not success:
                # If we didn't succeed, still add a placeholder score 
                score_list.append(0.0)
        
        predicted_target_class = int(np.argmax(score_list))
        max_score = float(np.max(score_list)) if score_list else 0.0

        print(f"model_name: {model_name}, predicted_target_class: {predicted_target_class}, max_score: {max_score}, scores: {score_list}\n\n")
        
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(f"model_name: {model_name}, predicted_target_class: {predicted_target_class}, max_score: {max_score}, scores: {score_list}\n\n")
        
        return predicted_target_class, score_list, trigger_list[predicted_target_class] if trigger_list else None 