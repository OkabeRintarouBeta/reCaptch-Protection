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

def predict_tile(tile_path):
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

        print(f"Predict image:{tile_path}\nNow prediction: {max_prob_class_name} with confidence {max_prob_confidence.item():.4f}")

# Test the function with the specified model and image path
# predict_tile("tlight-attack81-eps0.2-n1.png")
predict_tile("tlight-attack81-eps0.2-n20.png")
# predict_tile("tlight-targeted-attack81-to-Other-eps0.2-n1.png") 
predict_tile("tlight-targeted-attack81-to-Other-eps0.2-n20.png")    
# predict_tile("tlight-attack81-noni-eps0.2.png")
