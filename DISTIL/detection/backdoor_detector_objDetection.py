"""
Main backdoor detection pipeline for object detection models.

This module implements the detection pipeline for both SSD and Faster R-CNN models,
identifying potential backdoors by generating triggers that can manipulate model predictions.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Tuple, Dict, Optional, Union, Any

# Update imports to use existing project structure
import sys
if __name__ == "__main__":
    # When run as script, use absolute imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.append(project_root)
    from DISTIL.DISTIL.configs.config import (
        DEVICE,
        DEFAULT_ROOT_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_OUTPUT_FILE,
        DEFAULT_GUIDANCE_SCALE,
        DEFAULT_TIMESTEPS,
        DEFAULT_ARCHITECTURE,
        TRIGGER_CONFIDENCE_THRESHOLD
    )
    from DISTIL.DISTIL.utils.metrics import get_model_state
    from DISTIL.DISTIL.models.loader import load_detection_model, get_classifier_layer
    from DISTIL.DISTIL.models.similarity import greedy_class_farthest
    from DISTIL.DISTIL.utils.transforms import transform_image, transform_image_tensor
    # from DISTIL.DISTIL.detection.trigger_generator import (
    #     SSDTriggerGenerator,
    #     FRCNNTriggerGenerator,
    #     trigger_evaluation
    # )
else:
    from ..configs.config import (
    DEVICE,
    DEFAULT_ROOT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_TIMESTEPS,
    DEFAULT_ARCHITECTURE,
    TRIGGER_CONFIDENCE_THRESHOLD
)
from DISTIL.DISTIL.detection.trigger_generator import (
SSDTriggerGenerator,
FRCNNTriggerGenerator,
trigger_evaluation
)
# Global configuration
ARCHITECTURE = os.getenv("TROJDET_ARCHITECTURE", DEFAULT_ARCHITECTURE)
os.environ["TROJDET_ARCHITECTURE"] = ARCHITECTURE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.getenv("TROJDET_ROOT_DIR", DEFAULT_ROOT_DIR)
OUTPUT_DIR = os.getenv("TROJDET_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
OUTPUT_FILE = os.getenv("TROJDET_OUTPUT_FILE", DEFAULT_OUTPUT_FILE)

def generate_image_with_classifier(
    classifier: torch.nn.Module,
    timestep: int,
    guidance_scale: float,
    target_label: int,
    target_image: torch.Tensor,
    source_labels: Union[int, List[int]],
    ssd_model: Optional[torch.nn.Module] = None,
    detection_model: Optional[torch.nn.Module] = None
) -> Tuple[torch.Tensor, float]:
    """
    Generate a trigger image using classifier guidance.
    
    Args:
        classifier: Classification head of the detection model
        timestep: Number of diffusion steps
        guidance_scale: Strength of classifier guidance
        target_label: Target class to generate
        target_image: Starting image tensor
        source_labels: Original class(es) to move away from
        ssd_model: SSD model (if applicable)
        detection_model: Detection model (if applicable)
        
    Returns:
        Tuple of (generated_trigger, confidence_score)
    """
    ARCHITECTURE=os.environ['TROJDET_ARCHITECTURE']
    model = ssd_model if ssd_model is not None else detection_model
    
    # Determine if we're using FRCNN based on model structure or architecture setting
    is_frcnn = (
        (model is not None and hasattr(model, "model") and hasattr(model.model, "roi_heads")) or 
        ARCHITECTURE.lower() == "frcnn"
    )
    
    if is_frcnn:
        generator = FRCNNTriggerGenerator(device=DEVICE)
        return generator.generate_trigger(
            classifier=classifier,
            timestep=timestep,
            guidance_scale=guidance_scale,
            target_label=target_label,
            target_image=target_image,
            source_labels=source_labels,
            detection_model=model
        )
    else:
        generator = SSDTriggerGenerator(device=DEVICE)
        return generator.generate_trigger(
            classifier=classifier,
            timestep=timestep,
            guidance_scale=guidance_scale,
            target_label=target_label,
            target_image=target_image,
            source_labels=source_labels,
            ssd_model=model
        )


def process_detection_box(
    box_index: int,
    box_coords: Tuple[int, int, int, int],
    confidence: float,
    label: int,
    classifier_head: torch.nn.Module,
    label_mapping: Dict[int, int],
    base_tensor: torch.Tensor,
    images: torch.Tensor,
    detection_model: torch.nn.Module,
    triggers_dir: str,
    box_confidence_threshold: float = 0.5
) -> Optional[Tuple[int, Tuple[int, int, int, int], int, int, float, float, str, float]]:
    """
    Process a single detection box and generate a trigger if appropriate.
    
    Args:
        box_index: Index of the box in detection results
        box_coords: Coordinates (x1, y1, x2, y2) of the box
        confidence: Detection confidence score
        label: Detected class label
        classifier_head: Detection model's classifier
        label_mapping: Mapping from source to target labels
        base_tensor: Original image tensor
        images: Batch of images for evaluation
        detection_model: Detection model
        triggers_dir: Directory to save generated triggers
        box_confidence_threshold: Minimum confidence to process a box
        
    Returns:
        Box result data or None if skipped. Format:
        (box_index, coords, src_label, tgt_label, detection_confidence, 
         trigger_confidence, trigger_filepath, trigger_score)
    """
    ARCHITECTURE=os.environ['TROJDET_ARCHITECTURE']
    if confidence < box_confidence_threshold:
        return None
        
    # Extract valid box coordinates
    x1, y1, x2, y2 = box_coords
    x1c, y1c = max(0, x1), max(0, y1) 
    x2c, y2c = min(300, x2), min(300, y2)
    if x2c <= x1c or y2c <= y1c:
        return None
        
    src_label = int(label)
    if src_label not in label_mapping:
        return None
        
    tgt_label = label_mapping[src_label]
    patch = base_tensor[:, y1c:y2c, x1c:x2c].clone()
    # print(f"      [Box {box_index}] label={src_label}, far_label={tgt_label}, score={confidence:.2f}")
    
    # Generate trigger with adaptive guidance scale
    initial_guidance = 50 if ARCHITECTURE.lower() == 'ssd' else 25
    trigger = None
    trigger_confidence = 0.0
    trigger_score = 0.0
    
    for iteration in range(2):
        # print(f"      Iteration {iteration} with guid_scale = {initial_guidance}")
        trigger, trigger_confidence = generate_image_with_classifier(
            classifier=classifier_head,
            timestep=100 if ARCHITECTURE.lower() == 'ssd' else 50,
            guidance_scale=initial_guidance,
            target_label=tgt_label,
            target_image=patch,
            source_labels=src_label,
            detection_model=detection_model
        )
        
        if trigger_confidence > TRIGGER_CONFIDENCE_THRESHOLD or iteration == 1:
            # print(f"      Target {tgt_label}, Source {src_label}")
            trigger_score = trigger_evaluation(
                classifier=classifier_head,
                trigger=trigger.detach().cpu(),
                images=images,
                pred_labels=tgt_label if ARCHITECTURE.lower() == 'ssd' else None,
                pred_label=None if ARCHITECTURE.lower() == 'ssd' else tgt_label,
                ssd_model=detection_model if ARCHITECTURE.lower() == 'ssd' else None,
                detection_model=detection_model if ARCHITECTURE.lower() != 'ssd' else None
            )
            # print(f"      trigger_confidence = {trigger_confidence}, trigger_score = {trigger_score}")
            break
        else:
            initial_guidance *= 2.5 if ARCHITECTURE.lower() == 'ssd' else 3.5
    
    # print(f"      -> Final trigger confidence on label {tgt_label}: {trigger_confidence:.4f}")
    
    # Ensure trigger has correct dimensions and save
    if trigger.ndim == 4 and trigger.shape[0] == 1:
        trigger = trigger[0]
    trigger_img = transforms.ToPILImage()(trigger.detach().cpu())
    trigger_filename = f"box{box_index}_src{src_label}_to_{tgt_label}.png"
    trigger_filepath = os.path.join(triggers_dir, trigger_filename)
    trigger_img.save(trigger_filepath)
    # print(f"      -> Saved trigger patch to '{trigger_filepath}'")
    
    # Return all relevant information including both confidence and score
    return (box_index, (x1c, y1c, x2c, y2c), src_label, tgt_label, confidence, 
            trigger_confidence, trigger_filepath, trigger_score)


def process_model_folder(
    model_path: str,
    output_file: str = OUTPUT_FILE
) -> Optional[Tuple[str, str, List[float], float]]:
    """
    Process a single model folder to detect potential backdoors.
    
    Args:
        model_path: Path to the model folder
        output_file: File to write detailed results
        
    Returns:
        Model results summary or None if skipped
    """
    ARCHITECTURE=os.environ['TROJDET_ARCHITECTURE']
    folder_name = os.path.basename(model_path)
    ckpt_file = os.path.join(model_path, "model.pt")
    
    if not os.path.isfile(ckpt_file):
        # print(f"[SKIP] No model.pt in {model_path}.")
        return None
        
    # print(f"\n=== Attempting to load detection model from '{ckpt_file}' ===")
    detection_model = load_detection_model(ckpt_file, architecture=ARCHITECTURE)
    if detection_model is None:
        # print(f"[SKIP] Checkpoint '{ckpt_file}' is not recognized as {ARCHITECTURE}. Moving on.")
        return None
    # print("  -> Detection model loaded successfully.")
    
    # Get classifier head for detection model
    classifier_head = get_classifier_layer(detection_model)
    # print("  -> Computing farthest labels with 'greedy_class_farthest'...")
    far_list = greedy_class_farthest(detection_model)
    far_map = {i: j for (i, j, sim_val) in far_list}
    # print(f"  -> Found {len(far_map)} label mappings.")
    
    model_state = get_model_state(model_path)
    # print(f"  -> Model state: {model_state}")
    
    # Create output directory for triggers
    triggers_out_dir = os.path.join(
        os.path.join(OUTPUT_DIR, folder_name),
        "generated_triggers"
    )
    os.makedirs(triggers_out_dir, exist_ok=True)
    
    # Check for clean example data
    clean_data_dir = os.path.join(model_path, "clean-example-data")
    if not os.path.isdir(clean_data_dir):
        # print(f"[SKIP] No 'clean-example-data' in {model_path}.")
        return None
        
    # Get clean image files
    image_files = [
        os.path.join(clean_data_dir, fname)
        for fname in sorted(os.listdir(clean_data_dir))
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        # print(f"[SKIP] No clean example images found in '{clean_data_dir}'.")
        return None
        
    model_score_list = []
    results_list = []
    
    # Process a subset of images (adjust based on architecture)
    image_limit = 1 if ARCHITECTURE.lower() == 'frcnn' else 4
    
    for sample_image_path in image_files[:image_limit]:
        # print(f"  -> Processing image: '{sample_image_path}'")
        cv_image = cv2.imread(sample_image_path)
        if cv_image is None:
            # print(f"[SKIP] Cannot load '{sample_image_path}' via OpenCV.")
            continue
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        t224 = transform_image_tensor(cv_image)  # Expected shape: [1, 3, 224, 224]
        pil_224 = transforms.ToPILImage()(t224.squeeze(0))
        pil_300 = transforms.Resize((300, 300))(pil_224)
        in_300 = transforms.ToTensor()(pil_300).unsqueeze(0).to(DEVICE)
        images = in_300
        
        # Run detection on preprocessed image
        with torch.no_grad():
            if ARCHITECTURE.lower() == 'frcnn' and hasattr(detection_model.model, "roi_heads"):
                detections = detection_model([in_300.squeeze(0)])
            else:
                detections = detection_model(in_300)
        
        detection = detections[0]
        boxes = detection["boxes"].cpu().numpy()
        scores = detection["scores"].cpu().numpy()
        labels = detection["labels"].cpu().numpy()
        
        # Set confidence threshold based on architecture
        confidence_threshold = 0.99 if ARCHITECTURE.lower() == 'frcnn' else 0.5
        # print(f"    -> Found {len(boxes)} boxes. Considering those with conf > {confidence_threshold}.")
        
        base_tensor = transforms.ToTensor()(pil_300).to(DEVICE)
        
        # Process each detected box
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            result = process_detection_box(
                box_index=i, 
                box_coords=box.astype(int),
                confidence=score,
                label=label,
                classifier_head=classifier_head,
                label_mapping=far_map,
                base_tensor=base_tensor,
                images=images,
                detection_model=detection_model,
                triggers_dir=triggers_out_dir,
                box_confidence_threshold=confidence_threshold
            )
            
            if result is not None:
                # Unpack the result properly, now including trigger_score
                box_index, coords, src_label, tgt_label, detection_confidence, trigger_conf, trigger_path, trigger_score = result
                results_list.append(result)
                # Store the trigger_score instead of trigger_conf
                model_score_list.append(trigger_score)
                
    # Write per-model results if any triggers were generated
    if model_score_list:
        avg_score = sum(model_score_list) / len(model_score_list)
        with open(output_file, "a") as f:
            f.write(f"\nModel folder: {folder_name}\n")
            f.write(f"  Model state: {model_state}\n")
            f.write(f"  Trigger scores: {model_score_list}\n")
            f.write(f"  Average trigger score: {avg_score:.4f}\n")
            for result in results_list:
                idx, coords, src, tgt, det_conf, trig_conf, trig_file, trig_score = result
                f.write(f"  Box#{idx} coords={coords}, label: {src}->{tgt}, detection_conf={det_conf:.2f}, "
                        f"trigger_conf={trig_conf:.4f}, trigger_score={trig_score:.4f}, trigger='{trig_file}'\n")
            f.write("\n")
        return (folder_name, model_state, model_score_list, avg_score)
    else:
        # print(f"  -> No bounding boxes >{confidence_threshold} or triggers generated in this model.")
        return None
def main_detection_trigger_pipeline() -> None:
    """
    Main pipeline for detecting backdoors in object detection models.
    
    This function iterates through model folders in the ROOT_DIR,
    processes each model to find potential backdoors, and generates
    a detailed report of the findings.
    """
    if not os.path.isdir(ROOT_DIR):
        # print(f"[ERROR] '{ROOT_DIR}' doesn't exist.")
        return

    overall_results = []

    # Process each model folder
    for folder_name in sorted(os.listdir(ROOT_DIR)):
        model_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(model_path):
            continue
            
        result = process_model_folder(model_path, OUTPUT_FILE)
        if result is not None:
            overall_results.append(result)

    # Generate summary report
    # print("\nFinished main_detection_trigger_pipeline!")
    summary_file = os.path.join(OUTPUT_DIR, "overall_trigger_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write("Model\tState\tAvg_Trigger_Score\tScores\n")
        for folder_name, state, scores_, avg in overall_results:
            sf.write(f"{folder_name}\t{state}\t{avg:.4f}\t{scores_}\n")


if __name__ == "__main__":
    main_detection_trigger_pipeline()
