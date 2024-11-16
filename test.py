from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load the model
model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# Set a higher epsilon for FGSM to increase noise
epsilon = 0.1  # Adjust this value as needed to increase noise

def predict_tile_with_fgsm(tile_path):
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

    # Use max probability as mock loss for FGSM
    if result.probs is not None:
        max_prob_index = result.probs.top1  # Index of the top class
        max_prob_confidence = result.probs.top1conf  # Confidence of the top class
        max_prob_class_name = result.names[max_prob_index]

        # Print the original prediction details
        print(f"Original prediction: {max_prob_class_name} with confidence {max_prob_confidence.item():.4f}")

        # Mock FGSM loss by summing over all elements in the `to_predict` tensor
        mock_loss = to_predict.sum()
        mock_loss.backward()  # Compute gradients for FGSM

        # Generate adversarial image by adding FGSM noise to the original size image
        with torch.no_grad():
            sign_data_grad = to_predict.grad.sign()
            perturbed_image = to_predict + epsilon * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Keep pixel values within [0, 1]

        # Perform prediction on the polluted (adversarial) image after resizing it to 640x640
        perturbed_image_resized = torch.nn.functional.interpolate(perturbed_image, size=(640, 640), mode="bilinear", align_corners=False)
        polluted_results = model(perturbed_image_resized)
        polluted_result = polluted_results[0]
        polluted_max_prob_index = polluted_result.probs.top1
        polluted_max_prob_class_name = polluted_result.names[polluted_max_prob_index]
        polluted_max_prob_confidence = polluted_result.probs.top1conf

        # Print the polluted prediction details
        print(f"Polluted prediction: {polluted_max_prob_class_name} with confidence {polluted_max_prob_confidence.item():.4f}")

        # Convert perturbed image back to a displayable format for visualization at its original size
        perturbed_image_np = perturbed_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255  # Rescale to [0, 255]
        perturbed_image_np = perturbed_image_np.astype(np.uint8)

        # Display the perturbed image at the original size
        cv2.imshow("Adversarial Image", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the original and polluted predictions, and the adversarial image array
        return {
            "original": [result.probs.data, max_prob_class_name, max_prob_index],
            "polluted": [polluted_result.probs.data, polluted_max_prob_class_name, polluted_max_prob_index],
            "perturbed_image": perturbed_image_np
        }

    else:
        print("No predictions were made.")
        return None

# Test the function with the specified model and image path
predict_tile_with_fgsm("data/Training/Bicycle/Bicycle (98).png")  # Replace with your image path
