from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2
import argparse

class ImprovedTargetedFGSM:

    def __init__(self, model_path, class_dict, epsilon=0.05, num_steps=1, decay_factor=0.9):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.class_dict = class_dict
        self.alpha = epsilon / num_steps
        self.decay_factor = decay_factor

        self.image = None
        self.adv_image = None

        # Load the model
        model = YOLO(model_path)
        model.export(format='torchscript')

        # Load the TorchScript model
        torch_model_path = model_path.replace('.pt', '.torchscript.pt')
        self.torch_model = torch.jit.load(torch_model_path)
        self.torch_model.eval()  # Set the model to evaluation mode
    
    def get_epsilon(self):
        return self.epsilon
    
    def get_num_steps(self):
        return self.num_steps
    
    def get_decay_factor(self):
        return self.decay_factor

    def get_label_from_path(self, tile_path):
        class_name = tile_path.split('/')[-2]
        return list(self.class_dict.keys())[list(self.class_dict.values()).index(class_name)]

    def generate_adversarial_example(self, tile_path, target_class="Other"):
        target_index = list(self.class_dict.keys())[list(self.class_dict.values()).index(target_class)]
        y_target = torch.tensor([target_index], dtype=torch.long)  # Target class tensor for cross-entropy loss

        # Load and preprocess the image
        tile = Image.open(tile_path).convert("RGB")
        to_predict = transforms.ToTensor()(tile).unsqueeze(0)  # Shape: [1, 3, H, W]
        self.image = to_predict
        to_predict.requires_grad = True

        # Initialize adversarial example and momentum
        x_adv = to_predict.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(x_adv)  # Momentum for gradients

        # Attack iterations
        current_alpha = self.alpha
        for step in range(self.num_steps):
            x_adv.requires_grad = True  # Enable gradient tracking
            logits = self.torch_model(x_adv)  # Forward pass through the model

            # Compute targeted loss
            loss = F.cross_entropy(logits, y_target)  # Minimize the loss for the target class
            self.torch_model.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate to compute gradients

            with torch.no_grad():
                # Accumulate momentum
                grad = x_adv.grad / torch.norm(x_adv.grad, p=1)  # Normalize gradients
                momentum = self.decay_factor * momentum + grad

                # Apply momentum-based FGSM update
                grad_sign = momentum.sign()
                x_adv = x_adv - current_alpha * grad_sign

                # Clip the perturbation to maintain imperceptibility
                perturbation = torch.clamp(x_adv - to_predict, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)

                # Decay step size
                current_alpha *= self.decay_factor

        self.adv_image = x_adv
        
        # Convert the adversarial image to numpy for visualization
        perturbed_image_np = x_adv.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255  # Rescale to [0, 255]
        perturbed_image_np = perturbed_image_np.astype(np.uint8)

        return perturbed_image_np

    def predict(self):
        polluted_logits = self.torch_model(self.adv_image)
        polluted_pred = torch.argmax(polluted_logits, dim=1).item()

        original_pred = torch.argmax(self.torch_model(self.image), dim=1).item()
        print(f"Original prediction: {self.class_dict[original_pred]}")
        print(f"Adversarial prediction: {self.class_dict[polluted_pred]}")
        return self.class_dict[original_pred], self.class_dict[polluted_pred]