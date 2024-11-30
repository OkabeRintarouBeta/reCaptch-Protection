from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from attack.fgsm_untargeted import UntargetedFGSM
from attack.fgsm_untargeted_improved import ImprovedUntargetedFGSM
from attack.fgsm_targeted import TargetedFGSM
from attack.fgsm_targeted_improved import ImprovedTargetedFGSM

# Load the model
model_path = 'models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# generate adversarial example
def generate_adversarial_example(attack_type, tile_path):
    if attack_type == "untargeted_fgsm":
        attack_model = UntargetedFGSM(model_path, model.names)
    elif attack_type == "untargeted_fgsm_improved":
        attack_model = ImprovedUntargetedFGSM(model_path, model.names)
    elif attack_type == "targeted_fgsm":
        attack_model = TargetedFGSM(model_path, model.names)
    elif attack_type == "targeted_fgsm_improved":
        attack_model = ImprovedTargetedFGSM(model_path, model.names)

    adversarial_example = attack_model.generate_adversarial_example(tile_path)
    new_path = "adversarial-" + tile_path.split('/')[-1]
    cv2.imwrite(new_path, adversarial_example)
    attack_model.predict()
    return adversarial_example

# Test the function with the specified model and image path
generate_adversarial_example("targeted_fgsm_improved", "data/Training/Bridge/Bridge (79).png")