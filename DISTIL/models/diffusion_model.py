"""
Diffusion model implementation for generating trigger patches.
"""

import torch
import torch.nn.functional as F
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.download import load_checkpoint
from DISTIL.DISTIL.utils.transforms import transform_image_tensor
from DISTIL.DISTIL.utils.visualization import show_diffusion_images as show_images

class GLIDEModel:
    """
    Class to manage the GLIDE model and diffusion objects.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, timestep=None, device=None):
        """
        Get or create a GLIDEModel instance.
        
        Args:
            timestep: Number of timesteps for the diffusion process
            device: Device to run the model on
            
        Returns:
            GLIDEModel instance
        """
        if cls._instance is None:
            cls._instance = GLIDEModel(timestep, device)
        elif timestep is not None and timestep != cls._instance.timestep:
            # Reinitialize if timestep has changed
            cls._instance = GLIDEModel(timestep, device)
        
        return cls._instance
    
    def __init__(self, timestep, device=None):
        """
        Initialize the GLIDE model and diffusion object.
        
        Args:
            timestep: Number of timesteps for the diffusion process
            device: Device to run the model on
        """
        # Determine device if not provided
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.timestep = timestep
        
        # Initialize model settings
        self.options = model_and_diffusion_defaults()
        self.options["use_fp16"] = (device.type == "cuda")
        self.options["timestep_respacing"] = str(timestep)
        
        # Create and initialize the model
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.model.eval()
        if device.type == "cuda":
            self.model.convert_to_fp16()
        self.model.to(device)
        
        try:
            self.model.load_state_dict(load_checkpoint("base", device))
        except:
            print("Warning: Failed to load GLIDE checkpoint. Using random weights.")
    
    def get_model_and_diffusion(self):
        """
        Get the model, diffusion and options.
        
        Returns:
            model: The GLIDE model
            diffusion: The diffusion object
            options: Model options
        """
        return self.model, self.diffusion, self.options


def generate_image_with_classifier(classifier, timestep, guidance_scale_, target_label, target_image, source_labels, source_images=None,add_noise=True,grad_scale_factor=1/7):
    """
    Generate a backdoor trigger using Glide text-to-image model guided by the classifier.
    
    Args:
        classifier: Classifier model to guide the diffusion process
        timestep: Number of diffusion timesteps to use
        guidance_scale_: Guidance scale for diffusion model
        target_label: Target class label for the backdoor
        target_image: Example image of the target class
        source_labels: Labels for source classes
        source_images: Example images from source classes
    
    Returns:
        samples: Generated trigger images
        prob_of_target: Confidence score for the target class
    """
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    
    # Create base model with the specified timestep for diffusion
    options = model_and_diffusion_defaults()  
    options["use_fp16"] = has_cuda
    options["timestep_respacing"] = str(timestep)
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint("base", device))
    
    # Sampling parameters
    prompt = " "
    batch_size = 1
    guidance_scale = guidance_scale_
    
    # Create text tokens for the model
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options["text_ctx"])
    
    # Create classifier-free guidance tokens (empty tokens)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
    
    # Prepare model kwargs
    model_kwargs = dict(
        tokens=torch.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=torch.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size, dtype=torch.bool, device=device
        ),
    )
    
    # Define classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    # Define conditional function for classifier guidance
    def cond_fn(x, t, classifier, guidance_scale):
        with torch.enable_grad():
            x_single = x[0:1].clone().requires_grad_(True)
            x0 = transform_image_tensor(x_single).requires_grad_(True)
            
            # Add noise to make the trigger more robust
            input_img=x0
            if add_noise==True:
                input_img = input_img+ 0.3 * torch.rand_like(x0).to(device)
            logits = classifier((input_img).clip(0, 1).to(device))
            probs = F.softmax(logits, dim=1)
            
            # Compute gradient to maximize target class and minimize source class
            grad1 = torch.autograd.grad((logits[:, target_label] - logits[:, source_labels]).mean(), x_single, retain_graph=True)[0]
            
            # L1 regularization to keep the trigger small/sparse
            L1_loss = x_single.norm(p=1)
            grad2 = torch.autograd.grad(L1_loss, x_single, retain_graph=True)[0]   
            
            # Return combined gradient: maximize target class while keeping trigger small
            return guidance_scale * grad1 - guidance_scale * grad2 * grad_scale_factor  # Hyperparameter ratio
    
    # Wrapper for the conditional function
    def cond_fn_wrapper(x, t, **kwargs):
        return cond_fn(x, t, classifier, guidance_scale)
    
    # Clear model cache
    model.del_cache()
    
    # Run the diffusion process
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn_wrapper,
    )[:batch_size]
    
    # Clear cache again
    model.del_cache()
    
    # Evaluate trigger effectiveness
    with torch.no_grad():
        logits = classifier(transform_image_tensor(samples).clip(0, 1))
        prob_of_target = F.softmax(logits, dim=1)[0, target_label]
    
    return samples.clip(0, 1), prob_of_target.item()

def get_classifier_guidance_model(timestep=50, device=None):
    """
    Get an instance of the GLIDE model for classifier guidance.
    
    Args:
        timestep: Number of timesteps for the diffusion process
        device: Device to run the model on
    
    Returns:
        model: The GLIDE model
        diffusion: The diffusion object
        options: Model options
    """
    glide_instance = GLIDEModel.get_instance(timestep, device)
    return glide_instance.get_model_and_diffusion() 