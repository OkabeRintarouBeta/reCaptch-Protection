from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# Parameters for Iterative FGSM
epsilon = 1e-1  # Total perturbation magnitude
num_iterations = 10  # Number of iterations
alpha = epsilon / num_iterations  # Step size per iteration


def predict_tile_with_fgsm(tile_path):
    """
    Applies iterative FGSM to generate adversarial examples.
    """

    # Perform iterative FGSM
    tile = Image.open(tile_path).convert("RGB")
    original_size = tile.size  # Save the original image size for later use

    # Convert the image to a tensor and add a batch dimension
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: (1, 3, original_height, original_width)
    to_predict.requires_grad = True  # Enable gradients for FGSM
    
    for _ in range(num_iterations):

        # Resize to model's input size (128x128) and make prediction
        resized_input = torch.nn.functional.interpolate(
            to_predict, size=(128, 128), mode="bilinear", align_corners=False
        )
        results = model(resized_input)
        result = results[0]

        # Mock loss using top probability (use actual loss if available)
        loss = -result.probs.data.sum() if result.probs is not None else 0

        model.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate to calculate gradients

        with torch.no_grad():
            # Add perturbation
            sign_data_grad = to_predict.grad.sign()
            perturbed_tensor = to_predict + alpha * sign_data_grad
            # Clip the perturbation to ensure it stays within the epsilon bound
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)

    # Resize perturbed image back to original size
    perturbed_image_resized = torch.nn.functional.interpolate(
        perturbed_tensor, size=(128, 128), mode="bilinear", align_corners=False
    )

    # Convert to numpy for saving
    perturbed_image_np = perturbed_image_resized.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Save the perturbed image
    cv2.imshow("Adversarial Image", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()





# Test the function with the specified model and image path
predict_tile_with_fgsm("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path()

