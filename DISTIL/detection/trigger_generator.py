"""
Trigger generation for backdoor detection in object detection models.

This module implements utilities for generating backdoor triggers for both SSD 
and Faster R-CNN object detection architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults
)
from glide_text2im.download import load_checkpoint
from typing import Union, List, Tuple, Optional

from ..configs.config import DEVICE, DEFAULT_ARCHITECTURE
from ..models.feature_extractor import (
    extract_features_from_ssd,
    extract_features_from_detection
)
from ..utils.transforms import transform_image_tensor

# Architecture configuration
ARCHITECTURE = DEFAULT_ARCHITECTURE


class TriggerGenerator:
    """Base class for generating backdoor triggers for object detection models."""
    
    def __init__(self, device: torch.device = None):
        """
        Initialize the trigger generator.
        
        Args:
            device: Device to run generation on (defaults to global DEVICE)
        """
        self.device = device or DEVICE
    
    def setup_diffusion_model(self, timestep: int) -> Tuple[nn.Module, nn.Module, dict]:
        """
        Setup and initialize the diffusion model.
        
        Args:
            timestep: Number of timesteps for the diffusion process
            
        Returns:
            Tuple of (diffusion_model, diffusion, options)
        """
        opts = model_and_diffusion_defaults()
        opts["use_fp16"] = (self.device.type == "cuda")
        opts["timestep_respacing"] = str(timestep)
        
        diffusion_model, diffusion = create_model_and_diffusion(**opts)
        diffusion_model.eval()
        
        if self.device.type == "cuda":
            diffusion_model.convert_to_fp16()
        
        diffusion_model.to(self.device)
        try:
            diffusion_model.load_state_dict(load_checkpoint("base", self.device))
        except Exception as e:
            print(f"Warning: Failed to load GLIDE checkpoint. Using random weights. Error: {e}")
        
        return diffusion_model, diffusion, opts
    
    def create_diffusion_inputs(
        self, diffusion_model: nn.Module, opts: dict, batch_size: int = 1
    ) -> dict:
        """
        Create inputs for the diffusion model including tokens and masks.
        
        Args:
            diffusion_model: The GLIDE diffusion model
            opts: Model options
            batch_size: Batch size for generation
            
        Returns:
            Dictionary with model kwargs
        """
        prompt = " "  # Empty prompt for classifier-free guidance
        full_batch_size = batch_size * 2
        
        # Create text tokens
        tokens = diffusion_model.tokenizer.encode(prompt)
        tokens, mask = diffusion_model.tokenizer.padded_tokens_and_mask(tokens, opts["text_ctx"])
        
        # Create classifier-free guidance tokens
        uncond_tokens, uncond_mask = diffusion_model.tokenizer.padded_tokens_and_mask([], opts["text_ctx"])
        
        # Model kwargs
        model_kwargs = {
            "tokens": torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device),
            "mask": torch.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=torch.bool, device=self.device),
        }
        
        return model_kwargs
    
    def create_model_fn(self, diffusion_model: nn.Module, guidance_scale: float):
        """
        Create model function for diffusion sampling.
        
        Args:
            diffusion_model: The GLIDE diffusion model
            guidance_scale: Guidance scale for classifier-free guidance
            
        Returns:
            Model function for diffusion sampling
        """
        def model_fn(x_t: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            out = diffusion_model(combined, ts, **kwargs)
            eps, rest = out[:, :3], out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        
        return model_fn
    
    def prepare_initial_image(
        self, target_image: torch.Tensor, init_size: Union[int, Tuple[int, int]], batch_size: int = 1
    ) -> torch.Tensor:
        """
        Prepare the initial image for diffusion.
        
        Args:
            target_image: Target image tensor
            init_size: Size to resize the target image to
            batch_size: Batch size
            
        Returns:
            Initial noise tensor
        """
        if target_image.ndim == 3:
            target_image = target_image.unsqueeze(0)
            
        if isinstance(init_size, int):
            init_size = (init_size, init_size)
            
        init_img = F.interpolate(
            target_image,
            size=init_size,
            mode="bilinear",
            align_corners=False
        )
        
        # Expand to full batch size
        init_noise = init_img.expand(batch_size * 2, -1, -1, -1)
        return init_noise


class SSDTriggerGenerator(TriggerGenerator):
    """Trigger generator for SSD models."""
    
    def create_cond_fn(
        self,
        classifier: nn.Module,
        target_label: int,
        source_labels: Union[int, List[int]],
        ssd_model: nn.Module,
        guidance_scale: float
    ):
        """
        Create conditional function for SSD classifier guidance.
        
        Args:
            classifier: SSD classification head
            target_label: Target label to generate
            source_labels: Source label(s) to move away from
            ssd_model: SSD model
            guidance_scale: Guidance scale
            
        Returns:
            Conditional function for guided diffusion
        """
        def cond_fn(
            x: torch.Tensor, t: torch.Tensor, classifier_: nn.Module, guidance_scale_: float
        ) -> torch.Tensor:
            with torch.enable_grad():
                x_single = x[:1].clone().requires_grad_(True)
                x0 = transform_image_tensor(x_single).requires_grad_(True)
                perturbed = (0.3 * torch.rand_like(x0) + x0).clamp(0, 1)
                features = extract_features_from_ssd(ssd_model, perturbed)
                logits = classifier_([features]).mean(dim=1)
                src_list = [source_labels] if isinstance(source_labels, int) else source_labels
                loss = (logits[:, target_label] - logits[:, src_list].mean(dim=1)).mean()
                grad1 = torch.autograd.grad(loss, x_single, retain_graph=True)[0]
                l1_loss = x_single.norm(p=1)
                grad2 = torch.autograd.grad(l1_loss, x_single, retain_graph=True)[0]
                return guidance_scale_ * grad1 - guidance_scale_ * (grad2 / 5)
        
        def cond_fn_wrapper(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
            return cond_fn(x, t, classifier, guidance_scale)
        
        return cond_fn_wrapper
    
    def generate_trigger(
        self,
        classifier: nn.Module,
        timestep: int,
        guidance_scale: float,
        target_label: int,
        target_image: torch.Tensor,
        source_labels: Union[int, List[int]],
        ssd_model: nn.Module
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate a trigger for SSD models.
        
        Args:
            classifier: SSD classification head
            timestep: Diffusion timesteps
            guidance_scale: Diffusion guidance scale
            target_label: Target label to generate
            target_image: Starting image patch
            source_labels: Source label(s) to move away from
            ssd_model: SSD model
            
        Returns:
            Tuple of (trigger_tensor, confidence)
        """
        # Setup diffusion model
        diffusion_model, diffusion, opts = self.setup_diffusion_model(timestep)
        
        # Create model inputs
        batch_size = 1
        model_kwargs = self.create_diffusion_inputs(diffusion_model, opts, batch_size)
        
        # Create model and conditional functions
        model_fn = self.create_model_fn(diffusion_model, guidance_scale)
        cond_fn = self.create_cond_fn(
            classifier, target_label, source_labels, ssd_model, guidance_scale
        )
        
        # Prepare initial image
        init_noise = self.prepare_initial_image(
            target_image, opts["image_size"], batch_size
        )
        
        # Sample from diffusion model
        diffusion_model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (batch_size * 2, 3, init_noise.shape[2], init_noise.shape[3]),
            noise=init_noise,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )[:batch_size]
        diffusion_model.del_cache()
        
        # Evaluate trigger confidence
        with torch.no_grad():
            features = extract_features_from_ssd(
                ssd_model, transform_image_tensor(samples).clamp(0, 1)
            )
            test_logits = classifier([features]).mean(dim=1)
            prob_target = torch.softmax(test_logits, dim=1)[0, target_label]
            
        return samples.clamp(0, 1), prob_target.item()


class FRCNNTriggerGenerator(TriggerGenerator):
    """Trigger generator for Faster R-CNN models."""
    
    def create_cond_fn(
        self,
        classifier: nn.Module,
        target_label: int,
        source_labels: Union[int, List[int]],
        detection_model: nn.Module,
        guidance_scale: float
    ):
        """
        Create conditional function for FRCNN classifier guidance.
        
        Args:
            classifier: FRCNN classification head
            target_label: Target label to generate
            source_labels: Source label(s) to move away from
            detection_model: FRCNN model
            guidance_scale: Guidance scale
            
        Returns:
            Conditional function for guided diffusion
        """
        def cond_fn(
            x: torch.Tensor, t: torch.Tensor, classifier_: nn.Module, guidance_scale_: float
        ) -> torch.Tensor:
            with torch.enable_grad():
                x_single = x[:1].clone().requires_grad_(True)
                x0 = transform_image_tensor(x_single).requires_grad_(True)
                perturbed = (0.3 * torch.rand_like(x0) + x0).clamp(0, 1)
                
                # Branch for FasterRCNN vs. SSD:
                if hasattr(classifier_, "cls_score"):
                    features = extract_features_from_detection(detection_model, perturbed)
                    pooled_features = F.adaptive_avg_pool2d(features, (7, 7))
                    box_head = detection_model.model.roi_heads.box_head
                    box_features = box_head(pooled_features.view(pooled_features.size(0), -1))
                    classifier_out = classifier_(box_features)
                    if isinstance(classifier_out, tuple):
                        classifier_out = classifier_out[0]
                    logits = classifier_out
                else:
                    features = extract_features_from_ssd(detection_model, perturbed)
                    logits = classifier_([features]).mean(dim=1)
                    
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                    
                src_list = [source_labels] if isinstance(source_labels, int) else source_labels
                loss = (logits[:, target_label] - logits[:, src_list].mean(dim=1)).mean()
                grad1 = torch.autograd.grad(loss, x_single, retain_graph=True)[0]
                l1_loss = x_single.norm(p=1)
                grad2 = torch.autograd.grad(l1_loss, x_single, retain_graph=True)[0]
                return guidance_scale_ * grad1 - guidance_scale_ * (grad2 / 5)
        
        def cond_fn_wrapper(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
            return cond_fn(x, t, classifier, guidance_scale)
        
        return cond_fn_wrapper
    
    def adjust_classifier_head(self, classifier: nn.Module, detection_model: nn.Module) -> nn.Module:
        """
        Adjust the classifier head if needed for FRCNN models.
        
        Args:
            classifier: Classifier head
            detection_model: Detection model
            
        Returns:
            Adjusted classifier head
        """
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        if hasattr(classifier, "cls_score") and classifier.cls_score.weight.shape[0] == 1:
            num_classes = 91  # Adjust as needed
            in_features = classifier.cls_score.in_features
            new_predictor = FastRCNNPredictor(in_features, num_classes)
            detection_model.model.roi_heads.box_predictor = new_predictor
            classifier = new_predictor
            print(f"Classifier head replaced to force {num_classes} classes.")
            
        return classifier
    
    def generate_trigger(
        self,
        classifier: nn.Module,
        timestep: int,
        guidance_scale: float,
        target_label: int,
        target_image: torch.Tensor,
        source_labels: Union[int, List[int]],
        detection_model: nn.Module
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate a trigger for FRCNN models.
        
        Args:
            classifier: FRCNN classification head
            timestep: Diffusion timesteps
            guidance_scale: Diffusion guidance scale
            target_label: Target label to generate
            target_image: Starting image patch
            source_labels: Source label(s) to move away from
            detection_model: Detection model
            
        Returns:
            Tuple of (trigger_tensor, confidence)
        """
        # Adjust classifier head if needed
        classifier = self.adjust_classifier_head(classifier, detection_model)
        
        # Setup diffusion model
        diffusion_model, diffusion, opts = self.setup_diffusion_model(timestep)
        
        # Create model inputs
        batch_size = 1
        model_kwargs = self.create_diffusion_inputs(diffusion_model, opts, batch_size)
        
        # Create model and conditional functions
        model_fn = self.create_model_fn(diffusion_model, guidance_scale)
        cond_fn = self.create_cond_fn(
            classifier, target_label, source_labels, detection_model, guidance_scale
        )
        
        # Prepare initial image
        init_noise = self.prepare_initial_image(
            target_image, opts["image_size"], batch_size
        )
        
        # Sample from diffusion model
        diffusion_model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (batch_size * 2, 3, init_noise.shape[2], init_noise.shape[3]),
            noise=init_noise,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )[:batch_size]
        diffusion_model.del_cache()
        
        # Evaluate trigger confidence
        with torch.no_grad():
            if hasattr(classifier, "cls_score"):
                features = extract_features_from_detection(
                    detection_model, transform_image_tensor(samples).clamp(0, 1)
                )
                pooled_features = F.adaptive_avg_pool2d(features, (7, 7))
                box_head = detection_model.model.roi_heads.box_head
                box_features = box_head(pooled_features.view(pooled_features.size(0), -1))
                classifier_out = classifier(box_features)
                if isinstance(classifier_out, tuple):
                    classifier_out = classifier_out[0]
                test_logits = classifier_out
            else:
                features = extract_features_from_ssd(
                    detection_model, transform_image_tensor(samples).clamp(0, 1)
                )
                test_logits = classifier([features]).mean(dim=1)
                if test_logits.dim() == 4:
                    test_logits = F.adaptive_avg_pool2d(test_logits, (1, 1)).squeeze(-1).squeeze(-1)
                    
            if test_logits.dim() == 1:
                test_logits = test_logits.unsqueeze(0)
                
            probs = torch.softmax(test_logits, dim=1)
            prob_target = probs[0, target_label]
            
        return samples.clamp(0, 1), prob_target.item()


def apply_trigger_to_image(
    original_image: torch.Tensor, trigger: torch.Tensor, bbox: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Overlay the trigger onto the original image at specified bounding box coordinates.

    Args:
        original_image: Tensor [C, H, W] of the original image.
        trigger: Tensor representing the trigger patch.
        bbox: Coordinates (x1, y1, x2, y2).
    
    Returns:
        The blended image with the trigger applied.
    """
    out_image = original_image.clone()
    if trigger.ndim == 4 and trigger.shape[0] == 1:
        trigger = trigger[0]
    x1, y1, x2, y2 = bbox
    trig_h, trig_w = y2 - y1, x2 - x1
    trig_resized = F.interpolate(trigger.unsqueeze(0), size=(trig_h, trig_w),
                                 mode="bilinear", align_corners=False)[0]
    out_image[:, y1:y2, x1:x2] = (out_image[:, y1:y2, x1:x2] + trig_resized).clamp(0, 1)
    return out_image


def trigger_evaluation(
    classifier: nn.Module,
    trigger: torch.Tensor,
    images: torch.Tensor,
    pred_labels: Optional[int] = None,
    pred_label: Optional[int] = None,
    ssd_model: Optional[nn.Module] = None,
    detection_model: Optional[nn.Module] = None
) -> float:
    """
    Evaluate the trigger by blending it with images and computing classification confidence.

    Args:
        classifier: The detection classification head.
        trigger: Trigger patch tensor.
        images: Batch of images tensor.
        pred_labels: Target label for SSD models.
        pred_label: Target label for FRCNN models.
        ssd_model: SSD model (if applicable).
        detection_model: Generic detection model (if applicable).
    
    Returns:
        The average softmax probability (score) for the target label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    trigger = trigger.to(device)
    
    # Resize trigger to match image size
    trigger_resized = F.interpolate(
        trigger.unsqueeze(0) if trigger.ndim == 3 else trigger,
        size=images.shape[2:],
        mode="bilinear",
        align_corners=False
    )
    
    # Blend trigger with images
    if trigger_resized.ndim == 4 and trigger_resized.shape[0] == 1 and images.shape[0] > 1:
        trigger_resized = trigger_resized.expand(images.shape[0], -1, -1, -1)
    blended = (images + trigger_resized).clamp(0, 1)
    
    # Determine model type and extract features
    model = ssd_model if ssd_model is not None else detection_model
    if hasattr(model, "model") and hasattr(model.model, "roi_heads"):
        features = extract_features_from_detection(model, blended)
    else:
        features = extract_features_from_ssd(model, blended)
    
    # Get softmax probabilities for target label
    with torch.no_grad():
        if hasattr(classifier, "cls_score") and hasattr(model.model, "roi_heads"):
            # FRCNN path
            pooled_features = F.adaptive_avg_pool2d(features, (7, 7))
            box_head = model.model.roi_heads.box_head
            box_features = box_head(pooled_features.view(pooled_features.size(0), -1))
            classifier_out = classifier(box_features)
            if isinstance(classifier_out, tuple):
                classifier_out = classifier_out[0]
            logits = classifier_out
            target = pred_label
        else:
            # SSD path
            logits = classifier([features]).mean(dim=1)
            target = pred_labels
            
        if logits.dim() == 4:
            logits = F.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        probs = F.softmax(logits, dim=1)
        score = probs[:, target].mean().item()
        
    # print("Score (Extracted Features):", score)
    return score


def generate_trigger_patch(
    classifier: nn.Module,
    guidance_scale: float,
    timestep: int,
    target_label: int,
    target_image: torch.Tensor,
    source_labels: Union[int, List[int]],
    detection_model: nn.Module
) -> Tuple[torch.Tensor, float]:
    """
    Generate a Trojan trigger patch using the appropriate generator based on model type.

    Args:
        classifier: Classification head of the detection model.
        guidance_scale: Diffusion hyperparameter.
        timestep: Diffusion hyperparameter.
        target_label: Label to generate.
        target_image: Starting image patch.
        source_labels: Source label(s) to move away from.
        detection_model: The detection model.

    Returns:
        Tuple of (trigger_tensor, confidence).
    """
    # Determine if we're using FRCNN based on model structure
    is_frcnn = hasattr(detection_model, "model") and hasattr(detection_model.model, "roi_heads")
    
    if is_frcnn or ARCHITECTURE.lower() == "frcnn":
        generator = FRCNNTriggerGenerator()
        return generator.generate_trigger(
            classifier, timestep, guidance_scale, target_label, 
            target_image, source_labels, detection_model
        )
    else:
        generator = SSDTriggerGenerator()
        return generator.generate_trigger(
            classifier, timestep, guidance_scale, target_label, 
            target_image, source_labels, detection_model
        )
