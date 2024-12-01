from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
from keras.models import load_model
from attack.fgsm_untargeted_improved import ImprovedUntargetedFGSM
from attack.fgsm_untargeted import UntargetedFGSM
from attack.fgsm_targeted import TargetedFGSM
from attack.fgsm_targeted_improved import ImprovedTargetedFGSM

# Constants
CLASSES = ["Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk", "Hydrant", "Motorcycle", "Other", "Palm", "Stairs", "Traffic Light"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic light']
MODEL_OPTION = 'yolo'
attack_type= 'targeted_fgsm'

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

def predict_tile(tile_path):
    model_path = 'models/YOLO_Classification/train4/weights/best.pt'
    model = YOLO(model_path)

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

    return max_prob_class_name,result.probs


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
                        
                    label = str(subdir)
                    if(label.lower()==predicted_class.lower()):
                        correct_count+=1
                        class_correct_count+=1

                    
                    total += 1
                    class_total += 1

                
            # Calculate and write class accuracy to a file
            if class_total > 0:
                class_accuracy = class_correct_count / class_total

                # print(subdir_path, class_accuracy)
                # with open(f"class_accuracies_{attack_type}.txt", "a") as f:
                #     f.write(f"Class: {subdir}, Accuracy: {class_accuracy:.5f}\n")

    print("correct count: ", correct_count," total: ", total)
    print("Accuracy: ", correct_count/total)
    
    # Write overall accuracy to a file
    with open(f"class_accuracies_{attack_type}.txt", "a") as f:
        f.write(f"Total Correct count: {correct_count}, Total: {total}, Accuracy: {correct_count/total:.5f}\n")
        f.write("\n-------------------\n")
    return correct_count,total



# Modify train and validation directory path to test different datasets
train_dir=f"attack/generated_data/{attack_type}/Training"
val_dir=f"attack/generated_data/{attack_type}/Validation"
correct_count_train,total_train = traverse_files(train_dir)
correct_count_val,total_val = traverse_files(val_dir)
# print("Training Accuracy: ", correct_count_train/total_train)
# print("train_count: ", total_train)
# print("validation Accuracy: ", correct_count_val/total_val)
# print("val_count: ", total_val)
# print("Total Accuracy: ", (correct_count_train+correct_count_val)/(total_train+total_val))