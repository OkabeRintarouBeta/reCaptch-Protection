from models.YOLO_Classification import predict  
from PIL import Image  
import numpy as np  
import os

# Constants
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic']

def predict_tile(image_path):
    """
    Predict the class of an image tile using the YOLO model.
    
    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - Predicted class and confidence score.
    """
    # Use the `predict` function from YOLO_Classification
    result = predict.predict_tile(image_path)  

    # Extract the predicted class and probabilities
    probabilities = result[0]
    predicted_class = result[1]
    predicted_class_idx = result[2]

    return predicted_class, probabilities[predicted_class_idx]


def traverse_files(folder_path):
    """
    Traverse all files in a folder and predict the class of each image.
    
    Args:
    - folder_path (str): Path to the folder containing the images.
    """

    total=0
    correct_count=0

    # Traverse all subdirectories in the folder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            # Traverse all files in the subdirectory
            for file in os.listdir(subdir_path):
                # Check if the file is an image
                if file.endswith(".png") or file.endswith(".jpg"):
                    # Predict the class of the image
                    image_path = os.path.join(subdir_path, file)
                    predicted_class, confidence = predict_tile(image_path)
                    # The subdir name is the label
                    # print(subdir)
                    label = str(subdir)
                    if(label==predicted_class):
                        correct_count+=1
                    # print(f"Image: {file}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}, Label: {label}")
                    total+=1

    print("Accuracy: ", correct_count/total)

# image_path = "../recaptcha-dataset/Training/Bicycle/Bicycle (76).png" 
# predicted_class, confidence = predict_tile(image_path)
# print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

train_dir="recaptcha-dataset/Training"
traverse_files(train_dir)


