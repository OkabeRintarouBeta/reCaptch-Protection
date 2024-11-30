from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2

class TargetedFGSM:
    def __init__(self, model_path, class_dict, epsilon=0.05, num_steps=1):
        self.model_path = model_path
        self.class_dict = class_dict
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = epsilon / num_steps

        self.image = None
        self.adv_image = None

        # Load the model
        self.model = YOLO(model_path)
        self.model.export(format='torchscript')

        # Load the TorchScript model
        torchscript_model_path = model_path.replace('.pt', '.torchscript.pt')
        self.torch_model = torch.jit.load(torchscript_model_path)
        self.torch_model.eval()  # Set the model to evaluation mode
    
    def get_epsilon(self):
        return self.epsilon
    
    def get_num_steps(self):
        return self.num_steps

    def get_label_from_path(self, tile_path):
        class_name = tile_path.split('/')[-2]
        return list(self.class_dict.keys())[list(self.class_dict.values()).index(class_name)]

    def generate_adversarial_example(self, tile_path, target_class="Other"):
        target_index = list(self.class_dict.keys())[list(self.class_dict.values()).index(target_class)]
        y_target = torch.tensor([target_index], dtype=torch.long)

        tile = Image.open(tile_path).convert("RGB")
        to_predict = transforms.ToTensor()(tile).unsqueeze(0)
        self.image = to_predict
        to_predict.requires_grad = True

        x_adv = to_predict.clone().detach().requires_grad_(True)

        for step in range(self.num_steps):
            logits = self.torch_model(x_adv)
            loss = F.cross_entropy(logits, y_target)
            self.torch_model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv - self.alpha * grad_sign
                perturbation = torch.clamp(x_adv - to_predict, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)
                x_adv.requires_grad_(True)

        self.adv_image = x_adv

        perturbed_image_np = x_adv.detach().squeeze().permute(1, 2, 0).cpu().numpy() * 255
        perturbed_image_np = perturbed_image_np.astype(np.uint8)

        return perturbed_image_np

    def predict(self):
        polluted_logits = self.torch_model(self.adv_image)
        polluted_pred = torch.argmax(polluted_logits, dim=1).item()

        original_pred = torch.argmax(self.torch_model(self.image), dim=1).item()
        print(f"Original prediction: {self.class_dict[original_pred]}")
        print(f"Adversarial prediction: {self.class_dict[polluted_pred]}")
        return self.class_dict[original_pred], self.class_dict[polluted_pred]