from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2

class UntargetedFGSM:
    def __init__(self, model_path, class_dict, epsilon=0.2, num_steps=1):
        self.model_path = model_path
        self.class_dict = class_dict
        self.epsilon = epsilon 
        self.num_steps = num_steps  
        self.alpha = epsilon / num_steps  

        # Load the model
        self.model = YOLO(model_path)
        self.model.export(format='torchscript')

        torchscript_model_path = model_path.replace('.pt', '.torchscript.pt')
        self.torch_model = torch.jit.load(torchscript_model_path)
        self.torch_model.eval()

    def get_epsilon(self):
        return self.epsilon
    
    def get_num_steps(self):
        return self.num_steps
        

    def get_label_from_path(self, tile_path):
        class_name = tile_path.split('/')[-2]  # Get the second-to-last component of the path
        return list(self.class_dict.keys())[list(self.class_dict.values()).index(class_name)]

    def generate_adversarial_example(self, tile_path):
        y_index = self.get_label_from_path(tile_path)
        y = torch.tensor([y_index], dtype=torch.long)  # Target class tensor for cross-entropy loss

        # Load and preprocess the image
        tile = Image.open(tile_path).convert("RGB")
        to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
        to_predict.requires_grad = True

        # Initialize adversarial example
        x_adv = to_predict.clone().detach().requires_grad_(True)

        for step in range(self.num_steps):
            # Forward pass through the TorchScript model
            logits = self.torch_model(x_adv)  # Logits are raw predictions from the model
            # Compute the adversarial loss
            loss = -F.cross_entropy(logits, y)  # Negate to maximize loss
            self.torch_model.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate the loss

            # Perform FGSM-like update
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()  # Compute the sign of the gradient
                x_adv = x_adv + self.alpha * grad_sign  # Apply perturbation
                perturbation = torch.clamp(x_adv - to_predict, min=-self.epsilon, max=self.epsilon)  # Clip perturbation
                x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)  # Clip the image
                x_adv.requires_grad_(True)  # Re-enable gradient tracking for the next iteration

        # Forward pass through the model with the perturbed input
        polluted_logits = self.torch_model(x_adv)
        polluted_pred = torch.argmax(polluted_logits, dim=1).item()  # Predicted class index

        # Original prediction details
        original_pred = torch.argmax(self.torch_model(to_predict), dim=1).item()  # Original class index
        print(f"Original prediction: {self.class_dict[original_pred]}")
        print(f"Adversarial prediction: {self.class_dict[polluted_pred]}")
        # Convert the adversarial image to numpy for visualization
        perturbed_image_np = x_adv.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255  # Rescale to [0, 255]
        perturbed_image_np = perturbed_image_np.astype(np.uint8)

        return perturbed_image_np

# Test the class with the specified model and image path
# model_path = '../models/YOLO_Classification/train4/weights/best.pt'
# torchscript_model_path = 'models/YOLO_Classification/train4/weights/best.torchscript.pt'
# fgsm_attack = FGSMAttack(model_path, torchscript_model_path)
# fgsm_attack.predict_tile_with_fgsm("data/Training/Traffic Light/Tlight (81).png")  # Replace with your image path
