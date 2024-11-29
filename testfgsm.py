from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
from keras.models import load_model

# Load the model
model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 11: 'Stairs', 12: 'Traffic Light'}

# Set a higher epsilon for FGSM to increase noise
epsilon = 1  # Adjust this value as needed to increase noise
num_steps = 10 
def predict_tile_with_fgsm(tile_path):
    model.eval()
    y = torch
    for _ in range(num_steps):
        output = model(x_adv)
        loss = -F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Use .grad to access the gradient, and .sign() to get the sign of the gradient
            x_adv_update = x_adv - alpha * x_adv.grad.sign()
            # Calculate the perturbation (eta) and apply clipping
            eta = torch.clamp(x_adv_update - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + eta, min=0, max=255).detach_().requires_grad_(True)

    # Load the image at its original size and preprocess it
    tile = Image.open(tile_path).convert("RGB")
    original_size = tile.size  # Save the original image size for later use

    # Convert the image to a tensor and add a batch dimension
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: (1, 3, original_height, original_width)
    to_predict.requires_grad = True  # Enable gradients for FGSM

    # Perform the original prediction
    resized_input = torch.nn.functional.interpolate(to_predict, size=(128, 128), mode="bilinear", align_corners=False)
    results = model(resized_input)

    result = results[0]

    # Get the class names
    class_names = result.names
    print(class_names)
    # Get the probabilities
    probabilities = result.probs.data
    print(probabilities)

    # Get the class with the highest probability
    max_prob_index = result.probs.top1
    max_prob_class_name = class_names[max_prob_index] 
    print("max_idx:" + str(max_prob_index))
    print("output: " + max_prob_class_name)


def iterative_fgsm_untargeted(model, x, y, epsilon, alpha, num_steps=20):
    model.eval()
    # Convert numpy arrays to PyTorch tensors
    x_adv = x.clone().detach().requires_grad_(True).to(torch.float)
    y_true = y.clone().detach().to(torch.long)

    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        output = model(x_adv)
        loss = -F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Use .grad to access the gradient, and .sign() to get the sign of the gradient
            x_adv_update = x_adv - alpha * x_adv.grad.sign()
            # Calculate the perturbation (eta) and apply clipping
            eta = torch.clamp(x_adv_update - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + eta, min=0, max=255).detach_().requires_grad_(True)

    return x_adv

# Test the function with the specified model and image path
predict_tile_with_fgsm("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path
