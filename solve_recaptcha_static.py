from models.YOLO_Classification import predict  
from PIL import Image  
import numpy as np  
import os
from keras.models import load_model

# Constants
CLASSES = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", "Motorcycle", "Other", "Palm", "Stairs", "Traffic Light"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic light']
MODEL_OPTION = 'yolo'

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

    baseline_model=load_model('models/Base_Line/first_model.h5')

    # Traverse all subdirectories in the folder
    for subdir in os.listdir(folder_path):
        class_total=0
        class_correct_count=0
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            # Traverse all files in the subdirectory
            for file in os.listdir(subdir_path):
                # Check if the file is an image
                if file.endswith(".png") or file.endswith(".jpg"):
                    # Predict the class of the image
                    image_path = os.path.join(subdir_path, file)

                    if(MODEL_OPTION=='baseline'):
                        predicted_class, confidence=predict_image(image_path, baseline_model)
                    else:
                        predicted_class, confidence = predict_tile(image_path)
                        
                    # The subdir name is the label
                    # print(subdir)
                    label = str(subdir)
                    if(label=="Mountain"):
                            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
                    if(label.lower()==predicted_class.lower()):
                        correct_count+=1
                        class_correct_count+=1
                    # print(f"Image: {file}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}, Label: {label}")
                    total += 1
                    class_total += 1

            # Calculate and write class accuracy to a file
            if class_total > 0:
                class_accuracy = class_correct_count / class_total
                with open("class_accuracies.txt", "a") as f:
                    f.write(f"Class: {subdir}, Accuracy: {class_accuracy:.2f}\n")

    print("correct count: ", correct_count," total: ", total)
    print("Accuracy: ", correct_count/total)
    
    # Write overall accuracy to a file
    with open("class_accuracies.txt", "a") as f:
        f.write(f"Total Correct count: {correct_count}, Total: {total}, Accuracy: {correct_count/total:.2f}\n")
        f.write("\n-------------------\n")
    return correct_count,total

# image_path = "../recaptcha-dataset/Training/Bicycle/Bicycle (76).png" 
# predicted_class, confidence = predict_tile(image_path)
# print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

train_dir="data/Training"
val_dir="data/Validation"
correct_count_train,total_train = traverse_files(train_dir)
correct_count_val,total_val = traverse_files(val_dir)
print("Training Accuracy: ", correct_count_train/total_train)
print("train_count: ", total_train)
print("validation Accuracy: ", correct_count_val/total_val)
print("val_count: ", total_val)
print("Total Accuracy: ", (correct_count_train+correct_count_val)/(total_train+total_val))

