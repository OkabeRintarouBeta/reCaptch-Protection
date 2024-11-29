from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

model_path = '../models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# Parameters for Iterative FGSM
epsilon = 1e-1  # Total perturbation magnitude
num_iterations = 10  # Number of iterations
alpha = epsilon / num_iterations  # Step size per iteration


def modify_image_iterative_fgsm(predict_tensor, image_path):
    predict_tensor.requires_grad = True

    # Perform the iterative FGSM attack
    for _ in range(num_iterations):
        # Forward pass and calculate the loss (using a mock loss function)
        resized_input = torch.nn.functional.interpolate(predict_tensor, size=(128, 128), mode="bilinear", align_corners=False) # Add batch dimension
        results = model(resized_input)  
        result = results[0]
        
        # Use max probability as mock loss for FGSM
        mock_loss = -result.probs.top1conf
        model.zero_grad()
        mock_loss.backward()  

        # Get the gradient sign and update the image
        sign_data_grad = predict_tensor.grad.sign()
        predict_tensor = predict_tensor + alpha * sign_data_grad  # Apply perturbation
        predict_tensor = torch.clamp(predict_tensor, 0, 1)  # Ensure pixel values are between 0 and 1

    # Resize perturbed image to model's input size (128x128)
    perturbed_image_resized = torch.nn.functional.interpolate(
        predict_tensor, size=(128, 128), mode="bilinear", align_corners=False
    )

    perturbed_image_np = perturbed_image_resized.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    # Save the perturbed image (convert to BGR for OpenCV compatibility)
    cv2.imwrite(image_path, cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))



def predict_tile_with_fgsm(tile_path, tile_label):
    """
    Predicts the label of a tile and applies FGSM if the prediction is correct.
    """
    tile = Image.open(tile_path).convert("RGB")
    original_size = tile.size  # Save the original image size for later use

    # Convert the image to a tensor and add a batch dimension
    to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: (1, 3, original_height, original_width)
    to_predict.requires_grad = True  # Enable gradients for FGSM

    # Perform the original prediction
    resized_input = torch.nn.functional.interpolate(to_predict, size=(128, 128), mode="bilinear", align_corners=False)
    results = model(resized_input)
    result = results[0]

    # Use top class confidence as a mock loss
    max_prob_class_name = "unknown"
    correctness = False
    if result.probs is not None:
        max_prob_index = result.probs.top1  # Index of the top class
        max_prob_confidence = result.probs.top1conf  # Confidence of the top class
        max_prob_class_name = result.names[max_prob_index]
        correctness = max_prob_class_name.lower() == tile_label.lower()

        print(f"Prediction: {max_prob_class_name} with confidence {max_prob_confidence.item():.4f}")
        print(f"Tile label: {tile_label}")

    else:
        print("No predictions were made.")

    return to_predict, correctness


if __name__ == "__main__":
    root_dir = "../data"

    for category in ["Training", "Validation"]:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, category)):
            for dirname in dirnames:
                print(dirname)
                if not os.path.exists("iterative-fgsm-images/" + category + "/" + dirname):
                    os.makedirs("iterative-fgsm-images/" + category + "/" + dirname)
                dir = os.path.join(dirpath, dirname)
                for filename in os.listdir(dir):
                    if filename.endswith(".png"):
                        predict_tensor, correctness = predict_tile_with_fgsm(
                            os.path.join(dir, filename), dirname
                        )

                        new_image_path = "iterative-fgsm-images/" + category + "/" + dirname + "/" + filename
                        modify_image_iterative_fgsm(predict_tensor, new_image_path)
                        # # If the original prediction is incorrect, save the original image
                        # if not correctness:
                        #     # Copy the original image
                        #     cv2.imwrite(new_image_path, cv2.imread(os.path.join(dir, filename)))
                        # else:
                        #     # Apply iterative FGSM to generate an adversarial example
                        #     modify_image_iterative_fgsm(predict_tensor, new_image_path)
                        #     predict_tensor1, correctness1 = predict_tile_with_fgsm(new_image_path, dirname)
