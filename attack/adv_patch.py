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

def add_adversarial_patch_with_saliency(tile_path, target_class="Other", patch_size=20, num_steps=20, alpha=0.1):
    """
    Add an adversarial patch to an image, avoiding important regions using saliency maps.
    """
    # Map the target class name to its index
    target_index = list(CLASS.keys())[list(CLASS.values()).index(target_class)]
    y_target = torch.tensor([target_index], dtype=torch.long)  # Target class tensor

    # Load and preprocess the image
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]

    # Find the saliency map and exclude salient regions
    saliency_map = generate_saliency_map(to_predict.clone(), torch_model)
    patch_location = exclude_salient_regions(saliency_map, patch_size)
    patch = torch.rand(1, 3, patch_size, patch_size, requires_grad=True)

  
    for step in range(num_steps):
        patched_image = to_predict.clone()

        # Overlay the patch on the image at the chosen location
        row, col = patch_location
        patched_image[:, :, row:row + patch_size, col:col + patch_size] = patch

        # Forward pass through the model
        logits = torch_model(patched_image)

        # Compute the targeted loss
        loss = F.cross_entropy(logits, y_target)  # Minimize the loss to the target class
        torch_model.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagate to compute gradients

        # Update the patch using gradients
        with torch.no_grad():
            patch_grad = patch.grad
            patch -= alpha * patch_grad.sign()  # Gradient descent step
            patch.clamp_(0, 1)  # Ensure patch values are within [0, 1]
            patch.grad.zero_()  # Reset gradients for the patch

    # Final patched image
    patched_image = to_predict.clone()
    row, col = patch_location
    patched_image[:, :, row:row + patch_size, col:col + patch_size] = patch

    # Forward pass with the final patched image
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
    cv2.imwrite(f"adversarial-patch-saliency-patchsize{patch_size}-n{num_steps}.png", 
                cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))

# Test the function
add_adversarial_patch_with_saliency("data/Training/Traffic Light/Tlight (81).png", target_class="Other", patch_size=40, alpha=0.1)