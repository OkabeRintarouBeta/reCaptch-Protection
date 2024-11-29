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

alpha = epsilon = 0.2 # perturbation magnitude

# Map the tile path to the tensor index using the CLASS dictionary
def get_label_from_path(tile_path):
    # Extract the class name from the path
    # Assumes the path format is "data/Training/<Class Name>/<File Name>.png"
    class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
    return list(CLASS.keys())[list(CLASS.values()).index(class_name)]

def predict_tile_with_fgsm_non_iterative(tile_path):
    y_index = get_label_from_path(tile_path)
    y = torch.tensor([y_index], dtype=torch.long)  # Target class tensor for cross-entropy loss

    # Load and preprocess the image
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
    to_predict.requires_grad = True

    # Forward pass through the TorchScript model
    logits = torch_model(to_predict)  # Logits are raw predictions from the model
    loss = -F.cross_entropy(logits, y)  # Negate to maximize loss
    torch_model.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagate the loss

    # Compute the adversarial perturbation in a single step
    with torch.no_grad():
        grad_sign = to_predict.grad.sign()  # Compute the sign of the gradient
        perturbation = epsilon * grad_sign  # Scale the perturbation
        x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)  # Apply and clip the adversarial image

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
    cv2.imwrite(f"tlight-attack81-noni-eps{epsilon}.png", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))

# Test the function with the specified model and image path
predict_tile_with_fgsm_non_iterative("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path