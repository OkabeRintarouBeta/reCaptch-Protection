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

CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 11: 'Stairs', 12: 'Traffic Light'}

# Set a higher epsilon for FGSM to increase noise
epsilon = 1 # Adjust this value as needed to increase noise
num_steps = 10 
alpha = 0.1

# Map the tile path to the tensor index using the CLASS dictionary
def get_label_from_path(tile_path):
    # Extract the class name from the path
    # Assumes the path format is "data/Training/<Class Name>/<File Name>.png"
    class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
    return list(CLASS.keys())[list(CLASS.values()).index(class_name)]

def predict_tile_with_fgsm(tile_path):
    y_index = get_label_from_path(tile_path)
    y = torch.tensor([y_index], dtype=torch.long)  # Ensure y is a long tensor for cross-entropy

    # Load the image and preprocess
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
    to_predict.requires_grad = True

    # Resize to model's input size
    resized_input = F.interpolate(to_predict, size=(128, 128), mode="bilinear", align_corners=False)
    x_adv = resized_input.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        results = model(x_adv)
        print("results")
        print(results)
        result = results[0]

        # Use the raw probabilities (avoid .data to retain gradients)
        probabilities = result.probs.data.unsqueeze(0)  # Add batch dimension, shape: [1, num_classes]
        probabilities.requires_grad = True

        # Compute adversarial loss
        loss = -F.cross_entropy(probabilities, y)  # Negate to maximize loss
        model.zero_grad()
        loss.backward()

        # Perform FGSM-like update
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv - alpha * grad_sign  # Apply perturbation
            perturbation = torch.clamp(x_adv - resized_input, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(resized_input + perturbation, min=0, max=1).detach_().requires_grad_(True)

    # Convert the adversarial image back to numpy for visualization
    perturbed_image_np = x_adv.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Display the perturbed image
    cv2.imshow("Adversarial Image", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


# Test the function with the specified model and image path
predict_tile_with_fgsm("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path
