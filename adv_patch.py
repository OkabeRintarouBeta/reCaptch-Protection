from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2

# Load the model
model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)
model.export(format='torchscript')

# Load the model
model_path = 'models/YOLO_Classification/train4/weights/best.torchscript.pt'
torch_model = torch.jit.load(model_path)
torch_model.eval()  # Set the model to evaluation mode

# Class dictionary
CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 
         6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 
         11: 'Stairs', 12: 'Traffic Light'}

# Map the tile path to the tensor index using the CLASS dictionary
def get_label_from_path(tile_path):
    # Extract the class name from the path
    # Assumes the path format is "data/Training/<Class Name>/<File Name>.png"
    class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
    return list(CLASS.keys())[list(CLASS.values()).index(class_name)]

def add_adversarial_patch(tile_path, target_class="Other", patch_size=50, num_steps=20, alpha=0.1):
    """
    Add an adversarial patch to an image to achieve a targeted attack.
    """
    # Map the target class name to its index
    target_index = list(CLASS.keys())[list(CLASS.values()).index(target_class)]
    y_target = torch.tensor([target_index], dtype=torch.long)  # Target class tensor

    # Load and preprocess the image
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]

    # Create an adversarial patch with random values
    patch = torch.rand(1, 3, patch_size, patch_size, requires_grad=True)  # Random initialization

    # Define the location for the patch (top-left corner in this example)
    patch_location = (0, 0)  # (row, col) of the top-left corner

    for step in range(num_steps):
        patched_image = to_predict.clone()

        # Overlay the patch on the image
        patched_image[:, :, patch_location[0]:patch_location[0] + patch_size,
                      patch_location[1]:patch_location[1] + patch_size] = patch

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
    patched_image[:, :, patch_location[0]:patch_location[0] + patch_size,
                  patch_location[1]:patch_location[1] + patch_size] = patch

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
    cv2.imwrite(f"adversarial-patch-{target_class}-eps{patch_size}-n{num_steps}.png", 
                cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))


# Test the function
add_adversarial_patch("data/Training/Traffic Light/Tlight (81).png", target_class="Other", patch_size=50)