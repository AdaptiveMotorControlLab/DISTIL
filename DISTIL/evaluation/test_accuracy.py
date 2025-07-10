import torch
from tqdm import tqdm

def test_model(model, test_dataset, batch_size, transform, device):
    """
    Evaluate a predefined PyTorch model on a test dataset and return the accuracy.
    
    Parameters:
        model (torch.nn.Module): Pretrained model to test.
        test_dataset (Dataset): Test dataset where each item is a tuple (PIL image, label).
        batch_size (int): Number of samples per batch.
        transform (callable): Transformation to apply to each PIL image before inference.
        device (torch.device or str): Device on which computations are run ("cpu" or "cuda").
        
    Returns:
        accuracy (float): The fraction of samples correctly classified.
    """
    original_mode = model.training

    model = model.to(device)
    
    model.eval()

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch_imgs = []
            batch_labels = []
            for j in range(i, min(i + batch_size, len(test_dataset))):
                pil_img, label = test_dataset[j]
                img = transform(pil_img)
                batch_imgs.append(img)
                batch_labels.append(label)
            
            batch_tensor = torch.stack(batch_imgs).to(device)
            outputs = model(batch_tensor)

            _, predicted = torch.max(outputs, 1)
            
            batch_labels_tensor = torch.tensor(batch_labels, dtype=predicted.dtype)
            correct_predictions += (predicted.cpu() == batch_labels_tensor).sum().item()

            total_samples += len(batch_labels)

    if original_mode:
        model.train()
    else:
        model.eval()

    accuracy = correct_predictions / total_samples
    return accuracy 