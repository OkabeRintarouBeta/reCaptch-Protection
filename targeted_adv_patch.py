from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2

# Load the fine-tune YOLO model and export it to TorchScript 
model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)
model.export(format='torchscript')

model_path = 'models/YOLO_Classification/train4/weights/best.torchscript.pt'
torch_model = torch.jit.load(model_path)
torch_model.eval()  # Set the model to evaluation mode

# Class dictionary
CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 
         6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 
         11: 'Stairs', 12: 'Traffic Light'}

def get_label_from_path(tile_path):
    '''
    Map the tile path to the tensor index using the CLASS dictionary
    - Path format: "data/Training/<Class Name>/<File Name>.png"
    '''
    class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
    return list(CLASS.keys())[list(CLASS.values()).index(class_name)]

def generate_saliency_map(image_tensor, model):
    """
    Generate a saliency map for the input image using the model.
    Args:
        image_tensor: Input image tensor (1, 3, H, W).
        model: The classification model.
    Returns:
        saliency_map: Saliency map highlighting important regions, higher value, higher importance.
    """
    # Enable gradient calculation for the input image
    image_tensor.requires_grad = True 
    logits = model(image_tensor) 
    pred_class = torch.argmax(logits, dim=1)  
    loss = F.cross_entropy(logits, pred_class)  
    model.zero_grad()  
    loss.backward()  

    # Compute the absolute gradient values as the saliency map
    saliency_map = image_tensor.grad.abs().squeeze().mean(dim=0) # (3, H, W) -> (H, W)
    return saliency_map

def exclude_salient_regions(saliency_map, patch_size):
    """
    Identify the region to place the patch by excluding salient regions.
    Args:
        saliency_map: Saliency map highlighting important regions.
        patch_size: Size of the adversarial patch.
    Returns:
        patch_location: Coordinates (row, col) to place the patch.
    """
    # Normalize the saliency map to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    # Compute the importance score for each possible patch location
    H, W = saliency_map.shape
    min_importance = float('inf')
    best_location = (0, 0)

    # Iterate over every possible patch location
    for row in range(0, H - patch_size + 1):  # Allow overlapping placements
        for col in range(0, W - patch_size + 1):
            # Compute the sum of saliency values within the patch region
            region_importance = saliency_map[row:row + patch_size, col:col + patch_size].sum()
            if region_importance < min_importance:  # Lower importance is better
                min_importance = region_importance
                best_location = (row, col)

    print(f"Best location: {best_location} with importance {min_importance}")            
    return best_location

def get_target_patch(target_tensor, target_saliency_map, patch_size):
    """
    Extract the most salient patch from the target image based on its saliency map.
    Args:
        target_tensor: Tensor representation of the target image (1, 3, H, W).
        target_saliency_map: Saliency map of the target image (H, W).
        patch_size: Size of the patch to extract.
    Returns:
        patch: Extracted salient patch (1, 3, patch_size, patch_size).
    """
    # Normalize the saliency map to [0, 1]
    target_saliency_map = (target_saliency_map - target_saliency_map.min()) / (target_saliency_map.max() - target_saliency_map.min())

    # Find the most salient region
    H, W = target_saliency_map.shape
    max_saliency = float('-inf')
    best_location = (0, 0)

    # Slide a window over the saliency map to find the most salient region
    for row in range(0, H - patch_size + 1):
        for col in range(0, W - patch_size + 1):
            region_saliency = target_saliency_map[row:row + patch_size, col:col + patch_size].sum()
            if region_saliency > max_saliency:  # Higher saliency is better
                max_saliency = region_saliency
                best_location = (row, col)

    # Extract the patch from the target image
    row, col = best_location
    patch = target_tensor[:, :, row:row + patch_size, col:col + patch_size]

    return patch


def add_targeted_adversarial_patch(tile_path, target_image_path, patch_size=20, alpha=0.1, num_steps=20):
    """
    Add a targeted adversarial patch to an image, aiming to misclassify it as the class of a target image.
    """
    # Map the input and target images to their respective classes
    y_index = get_label_from_path(tile_path)
    y = torch.tensor([y_index], dtype=torch.long)
    y_index_target = get_label_from_path(target_image_path)
    y_target = torch.tensor([y_index_target], dtype=torch.long)

    # Load and preprocess the input image
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)
    patch_tile = Image.open(target_image_path).convert("RGB")
    patch_tensor = transforms.ToTensor()(patch_tile).unsqueeze(0)

    # Generate saliency map and determine patch location
    saliency_map = generate_saliency_map(to_predict.clone(), torch_model)    
    patch_location = exclude_salient_regions(saliency_map, patch_size)

    # Use the most salient region of the target image as the patch
    target_salieny_map = generate_saliency_map(patch_tensor.clone(), torch_model)
    patch = get_target_patch(patch_tensor, target_salieny_map, patch_size)
    patch.requires_grad = True

    patched_image = to_predict.clone()

    # Overlay the patch on the image at the chosen location
    row, col = patch_location
    patched_image[:, :, row:row + patch_size, col:col + patch_size] = patch

    # Forward pass through the model
    logits = torch_model(patched_image)

    # Compute the targeted loss (minimize the loss for the target class)
    loss = F.cross_entropy(logits, y)  # Minimize loss for the target class
    torch_model.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagate to compute gradients

    # Update the patch using gradients
    with torch.no_grad():
        patch_grad = patch.grad
        patch -= alpha * patch_grad.sign()  # Gradient descent step
        patch.clamp_(0, 1)  # Ensure patch values are within [0, 1]
        patch.grad.zero_()  # Reset gradients for the patch

    # Apply the final patch
    patched_image = to_predict.clone()
    patched_image[:, :, row:row + patch_size, col:col + patch_size] = patch

    # Final forward pass with the patched image
    polluted_logits = torch_model(patched_image)
    polluted_pred = torch.argmax(polluted_logits, dim=1).item()

    # Original prediction details
    original_pred = torch.argmax(torch_model(to_predict), dim=1).item()
    print(f"Original prediction: {CLASS[original_pred]}")
    print(f"Adversarial prediction: {CLASS[polluted_pred]}")

    # Convert the patched image to numpy for visualization
    perturbed_image_np = patched_image.detach().squeeze().permute(1, 2, 0).numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Save the patched adversarial image
    cv2.imwrite(f"targeted-adversarial-patch-patchsize{patch_size}.png", 
                cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))

# Example Usage:
tile_path = "data/Training/Traffic Light/Tlight (81).png"  # Input image to attack
target_image_path = "data/Training/Bus/Bus (75).png"  # Target image specifying the target class
add_targeted_adversarial_patch(tile_path, target_image_path, patch_size=50, alpha=0.1, num_steps=20)

