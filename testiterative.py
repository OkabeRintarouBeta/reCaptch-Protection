from ultralytics import YOLO
import torch
import torch.nn.functional as F
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
    # Load the image and preprocess
    tile = Image.open(tile_path).convert("RGB")
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
    to_predict.requires_grad = True

    # Initialize adversarial example
    x_adv = to_predict.clone()

    # Original prediction
    resized_input = F.interpolate(x_adv, size=(128, 128), mode="bilinear", align_corners=False)
    results = model(resized_input)
    result = results[0]
    max_prob_index = result.probs.top1  # Index of the top class
    max_prob_confidence = result.probs.top1conf  # Confidence of the top class
    max_prob_class_name = result.names[max_prob_index]

    print(f"Original prediction: {max_prob_class_name} with confidence {max_prob_confidence.item():.4f}")

    # Perform iterative FGSM
    for _ in range(num_iterations):
        # Resize input for model
        resized_input = F.interpolate(x_adv, size=(128, 128), mode="bilinear", align_corners=False)
        results = model(resized_input)
        result = results[0]

        # Compute adversarial loss (maximize the confidence of an incorrect class)
        loss = -to_predict.sum() # Use the top-1 confidence as the loss
        model.zero_grad()
        loss.backward()

        # Update adversarial example
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign  # Apply perturbation
            print(x_adv)
            perturbation = torch.clamp(x_adv - to_predict, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)
            x_adv.requires_grad = True  # Re-enable gradient tracking for the next iteration

    # Polluted prediction
    polluted_results = model(x_adv)
    polluted_result = polluted_results[0]
    polluted_max_prob_index = polluted_result.probs.top1
    polluted_max_prob_class_name = polluted_result.names[polluted_max_prob_index]
    polluted_max_prob_confidence = polluted_result.probs.top1conf

    print(f"Polluted prediction: {polluted_max_prob_class_name} with confidence {polluted_max_prob_confidence.item():.4f}")

    # Convert the adversarial image back to numpy for visualization
    perturbed_image_np = x_adv.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Display the perturbed image
    cv2.imshow("Adversarial Image", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()




# Test the function with the specified model and image path
predict_tile_with_fgsm("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path()

