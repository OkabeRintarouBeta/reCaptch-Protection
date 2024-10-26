from models.YOLO_Classification import predict  
from PIL import Image  
import numpy as np  
import os
from tensorflow.keras.models import load_model

# Constants
BASELINE_CLASSES = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", "Motorcycle", "Other", "Palm", "Stairs", "Traffic Light"]
YOLO_CLASSES = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Mountain', 'Other', 'Palm', 'Traffic Light']
MODEL_OPTION = 'baseline'
# MODEL_OPTION = 'YOLO'

if MODEL_OPTION=='baseline':
    CLASS = BASELINE_CLASSES
    # initialize the model
    baseline_model = load_model('models/Base_Line/first_model.h5')
elif MODEL_OPTION=='YOLO':
    CLASS = YOLO_CLASSES

# initialize per-class counters
class_correct_counts = {cls: 0 for cls in CLASS}
class_total_counts = {cls: 0 for cls in CLASS}

def predict_image(image_path, model):
    """
    Predict the class of an image tile using a specified model.
    
    Args:
    - image_path (str): Path to the image file.
    - model_path (str): Path to the model file (.h5).
    
    Returns:
    - Predicted class and confidence score.
    """
    
    # Load and preprocess the image
    tile = Image.open(image_path)
    tile = tile.convert("RGB").resize((224, 224))
    image_array = np.array(tile) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(image_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_class_idx]
    
    return CLASSES[predicted_class_idx], confidence

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
                    image_path = os.path.join(subdir_path, file)
                    label = str(subdir)
                    # Predict the class of the image
                    if(MODEL_OPTION=='baseline'):
                        predicted_class, confidence=predict_image(image_path, baseline_model)
                    else:
                        predicted_class, confidence = predict_tile(image_path)

                    # Update label counts
                    if label in class_total_counts:
                        class_total_counts[label] += 1
                    # Check if the predicted class matches the label, and update correct counts
                    if label == predicted_class:
                        correct_count += 1
                        if label in class_correct_counts:
                            class_correct_counts[label] += 1
                    total += 1

    # Calculate and display per-class accuracy
    print("\nPer-Class Accuracy:")
    for cls in CLASS:
        class_accuracy = (class_correct_counts[cls] / class_total_counts[cls]) if class_total_counts[cls] > 0 else 0
        print(f"{cls}: {class_accuracy:.2f}")

    # Calculate overall accuracy
    accuracy = correct_count / total if total > 0 else 0
    return accuracy, total

# image_path = "../recaptcha-dataset/Training/Bicycle/Bicycle (76).png" 
# predicted_class, confidence = predict_tile(image_path)
# print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

# Paths to the training and validation datasets
train_dir="data/Training"
val_dir="data/Validation"

print("\n--- Training Evaluation ---")
train_accuracy, total_train = traverse_files(train_dir)
print("Training Accuracy: ", train_accuracy)

print("\n--- Validation Evaluation ---")
val_accuracy, total_val = traverse_files(val_dir)
print("Validation Accuracy: ", val_accuracy)

print("\nTotal Accuracy: ", (train_accuracy*total_train + val_accuracy*total_val)/(total_train+total_val))


