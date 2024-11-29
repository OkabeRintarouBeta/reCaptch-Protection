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

# Parameters for FGSM
epsilon = 0.2  # Total perturbation magnitude
num_steps = 20  # Number of iterations
alpha = epsilon / num_steps  # Step size per iteration

# Map the tile path to the tensor index using the CLASS dictionary
def get_label_from_path(tile_path):
    # Extract the class name from the path
    # Assumes the path format is "data/Training/<Class Name>/<File Name>.png"
    class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
    return list(CLASS.keys())[list(CLASS.values()).index(class_name)]

def improved_untargeted_fgsm(tile_path, decay_factor=0.9):
    """
    Improvements: Momentum-based updates and normalized gradients
    - Accumulates gradient information from previous steps to stabilize updates and improve attack efficiency.
    - Normalized gradients ensure smooth perturbations.

    Benefits:
	- Stability: Momentum-based updates reduce the risk of oscillations.
    """
    # Load and preprocess the image
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
    to_predict.requires_grad = True

    # Initialize adversarial example and momentum
    x_adv = to_predict.clone().detach().requires_grad_(True)
    momentum = torch.zeros_like(x_adv)  # Momentum for gradients

    # Attack iterations
    current_alpha = alpha
    for step in range(num_steps):
        x_adv.requires_grad = True  # Enable gradient tracking
        logits = torch_model(x_adv)  # Forward pass through the model

        # Compute untargeted loss (maximize the loss for the current prediction)
        original_pred = torch.argmax(torch_model(to_predict), dim=1)
        loss = -F.cross_entropy(logits, original_pred)  # Maximize the loss for the original prediction
        torch_model.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagate to compute gradients

        with torch.no_grad():
            # Accumulate momentum
            grad = x_adv.grad / torch.norm(x_adv.grad, p=1)  # Normalize gradients
            momentum = decay_factor * momentum + grad

            # Apply momentum-based FGSM update
            grad_sign = momentum.sign()
            x_adv = x_adv + current_alpha * grad_sign  # Update in the untargeted direction

            # Clip the perturbation to maintain imperceptibility
            perturbation = torch.clamp(x_adv - to_predict, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)

            # Decay step size
            current_alpha *= decay_factor

    # Forward pass through the model with the perturbed input
    polluted_logits = torch_model(x_adv)
    polluted_pred = torch.argmax(polluted_logits, dim=1).item()  # Predicted class index

    # Original prediction details
    original_pred = torch.argmax(torch_model(to_predict), dim=1).item()  # Original class index
    print(f"Original prediction: {CLASS[original_pred]}")
    print(f"Adversarial prediction: {CLASS[polluted_pred]}")
    
    # Convert the adversarial image to numpy for visualization
    perturbed_image_np = x_adv.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255  # Rescale to [0, 255]
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Save the adversarial image
    cv2.imwrite(f"improved-untargeted-attack81-eps{epsilon}-n{num_steps}.png", 
                cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))

# Test the function with the specified model and image path
improved_untargeted_fgsm("data/Training/Traffic Light/Tlight (81).png")