import os
import torch

def list_gpus():
    """
    List available GPUs and their details.
    
    Returns:
        device: The PyTorch device (cuda or cpu)
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU found.")
    else:
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def set_gpu(gpu_id=None):
    """
    Set the GPU to use.
    
    Args:
        gpu_id: ID of the GPU to use. If None, use the default.
        
    Returns:
        device: The PyTorch device (cuda or cpu)
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    return list_gpus()

if __name__ == "__main__":
    list_gpus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 