from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

model_path = '../models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# Set a higher epsilon for FGSM to increase noise
epsilon = 3e-1

def modify_image(predict_tensor, image_path):
    # Mock FGSM loss
    mock_loss = predict_tensor.sum()
    mock_loss.backward()  

    with torch.no_grad():
        sign_data_grad = predict_tensor.grad.sign()
        perturbed_image = predict_tensor + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Resize perturbed image to model's input size (640x640)
    perturbed_image_resized = torch.nn.functional.interpolate(
        perturbed_image, size=(640, 640), mode="bilinear", align_corners=False
    )

    
    perturbed_image_np = perturbed_image_resized.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    perturbed_image_np = perturbed_image_np.astype(np.uint8)

    cv2.imwrite(image_path, cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))


def predict_tile_with_fgsm(tile_path):
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
        # print(f" prediction: {max_prob_class_name} with confidence {max_prob_confidence.item():.4f}")

    else:
        print("No predictions were made.")
    
    return to_predict


if __name__ == "__main__":
    root_dir="../data"

    for category in ["Training", "Validation"]:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, category)):
            print(dirpath)
            for dirname in dirnames:
                print(dirname)
                if not os.path.exists("yolo8-gen-images/"+category+"/"+dirname):
                    os.makedirs("yolo8-gen-images/"+category+"/"+dirname)
                dir=os.path.join(dirpath, dirname)
                for filename in os.listdir(dir):
                    if filename.endswith(".png"):
                        predict_tensor=predict_tile_with_fgsm(os.path.join(dir, filename))
                        new_image_path="yolo8-gen-images/"+category+"/"+dirname+"/"+filename
                        modify_image(predict_tensor,new_image_path)  
                        predict_tile_with_fgsm(new_image_path)

    

