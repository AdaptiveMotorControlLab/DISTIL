import torch as th
import torch.nn.functional as F
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.download import load_checkpoint
from ..utils.transforms import transform_image_tensor
from ..utils.visualization import show_diffusion_images as show_images

class GLIDEUpsampleModel:
    """
    Class to manage the GLIDE upsampler model and diffusion objects.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, timestep=None, device=None):
        """
        Get or create a GLIDEUpsampleModel instance.
        
        Args:
            timestep: Number of timesteps for the diffusion process (default is 50 for upsampler)
            device: Device to run the model on
            
        Returns:
            GLIDEUpsampleModel instance
        """
        if cls._instance is None:
            cls._instance = GLIDEUpsampleModel(device)
        
        return cls._instance
    
    def __init__(self, device=None):
        """
        Initialize the GLIDE upsampler model and diffusion object.
        
        Args:
            device: Device to run the model on
        """
        # Determine device
        has_cuda = th.cuda.is_available()
        self.device = th.device("cpu" if not has_cuda else "cuda") if device is None else device
        
        # Initialize model settings
        self.options = model_and_diffusion_defaults_upsampler()
        self.options["use_fp16"] = has_cuda
        self.options["timestep_respacing"] = "50"  # Fixed timestep for upsampler
        
        # Create and initialize the model
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.model.eval()
        if has_cuda:
            self.model.convert_to_fp16()
        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint("upsample", self.device))
    
    def get_model_and_diffusion(self):
        """
        Get the model, diffusion and options.
        
        Returns:
            model: The GLIDE upsampler model
            diffusion: The diffusion object
            options: Model options
        """
        return self.model, self.diffusion, self.options


def run_diffusion_with_trigger(images_, classifier, target_label, source_label, guidance_scale=5.0, upsample_temp=0.997):
    """
    Run the diffusion process with the upsampler model guided by the classifier.
    
    Args:
        images_: Initial low-resolution images
        classifier: The classifier model to guide the diffusion
        target_label: Target label for the backdoor
        source_label: Source label to avoid
        guidance_scale: Scale factor for the classifier guidance
        upsample_temp: Temperature for the upsampling noise
        
    Returns:
        upsampled_image: The upsampled image
        trigger_confidence: Confidence of the trigger
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    # Initialize a list to store triggered information if condition is met
    triggered = []
    
    # Get or initialize the GLIDE upsampler model using the class
    glide_upsample_instance = GLIDEUpsampleModel.get_instance(device)
    model_up, diffusion_up, options_up = glide_upsample_instance.get_model_and_diffusion()
    
    # Prepare the initial image
    init_image = images_.to(device)
    
    # Define the conditional function for classifier guidance
    def cond_fn(x, t, classifier, guidance_scale):
        with th.enable_grad():
            x_single = x[0:1].clone().requires_grad_(True)
            x0 = transform_image_tensor(x_single).requires_grad_(True)
            logits = classifier((images_.cuda() + x0).clip(0,1).cuda())
            probs = F.softmax(logits, dim=1)
            grad1 = th.autograd.grad((logits[:, target_label] - logits[:, source_label]).mean(), 
                                     x_single, retain_graph=True)[0]
            L1_loss = x_single.norm(p=1)
            grad2 = th.autograd.grad(L1_loss, x_single, retain_graph=True)[0]
            if t[0].item() == 41:  # Store trigger information at step 41
                triggered.append(x)
                triggered.append(probs[0, target_label].item())
            return guidance_scale * grad1 - grad2

    def cond_fn_wrapper(x, t, **kwargs):
        return cond_fn(x, t, classifier, guidance_scale)
    
    # Set up model kwargs
    batch_size = 1
    tokens = model_up.tokenizer.encode("")
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(tokens, options_up['text_ctx'])
    model_kwargs = {
        "low_res": init_image,
        "tokens": th.tensor([tokens] * batch_size, device=device),
        "mask": th.tensor([mask] * batch_size, dtype=th.bool, device=device),
    }
    
    # Run the diffusion process
    model_up.del_cache()
    up_shape = (batch_size, 3, 224, 224)
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=None,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn_wrapper,
    )[:batch_size]
    model_up.del_cache()
    
    upsampled_image = transform_image_tensor(up_samples[0]).clip(0,1).detach().cpu()
    
    if triggered:
        upsampled_image= triggered[0]
        trigger_confidence = triggered[1]
    else:
        # Fallback: compute confidence on the upsampled image
        logits_final = classifier(upsampled_image.unsqueeze(0).cuda())
        trigger_confidence = F.softmax(logits_final, dim=1)[0, target_label].item()
    
    return upsampled_image, trigger_confidence
