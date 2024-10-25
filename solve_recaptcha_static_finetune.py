from models.YOLO_Classification import predict  
from PIL import Image  
import numpy as np  
import os

# Constants
YOLO_CLASSES = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Mountain', 'Other', 'Palm', 'Traffic']

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


def evaluate_model(folder_path):
    """
    Evaluate model accuracy by predicting class for images in a folder.

    Args:
    - folder_path (str): Path to the folder containing images.
    
    Returns:
    - accuracy (float): Accuracy of model on dataset.
    """

    total = 0
    correct_count = 0
    # Initialize per-class counters
    class_correct_counts = {cls: 0 for cls in YOLO_CLASSES}
    class_total_counts = {cls: 0 for cls in YOLO_CLASSES}

    # Traverse all subdirectories in the folder
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".png") or file.endswith(".jpg"):
                    image_path = os.path.join(subdir_path, file)
                    predicted_class, _ = predict_tile(image_path)
                    label = str(subdir)

                    # Update label counts
                    if label in class_total_counts:
                        class_total_counts[label] += 1
                    # print(class_total_counts)
                    # Check if the predicted class matches the label, and update correct counts
                    if label == predicted_class:
                        correct_count += 1
                        if label in class_correct_counts:
                            class_correct_counts[label] += 1
                    total += 1
                    # print(class_correct_counts)
    
    # Calculate overall accuracy
    accuracy = correct_count / total if total > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2f}")
    
    # Calculate and display per-class accuracy
    print("\nPer-Class Accuracy:")
    for cls in YOLO_CLASSES:
        class_accuracy = (class_correct_counts[cls] / class_total_counts[cls]) if class_total_counts[cls] > 0 else 0
        print(f"{cls}: {class_accuracy:.2f}")
    
    return accuracy


train_dir = "data/Training"
validation_dir = "data/Validation"

print("\n--- Training Evaluation ---")
train_accuracy = evaluate_model(train_dir)

print("\n--- Validation Evaluation ---")
validation_accuracy = evaluate_model(validation_dir)

print("\nFinal Report:")
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}")