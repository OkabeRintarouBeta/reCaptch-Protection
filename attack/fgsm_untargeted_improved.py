from ultralytics import YOLO
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  
import numpy as np  
import os
import cv2

class ImprovedUntargetedFGSM:
    def __init__(self, model_path, class_dict, epsilon=0.08, num_steps=1, decay_factor=0.9):
        self.model_path = model_path
        self.class_dict = class_dict
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = epsilon / num_steps
        self.decay_factor = decay_factor

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
    
    def get_decay_factor(self):
        return self.decay_factor

    def get_label_from_path(self, tile_path):
        class_name = tile_path.split('/')[-2]
        return list(self.class_dict.keys())[list(self.class_dict.values()).index(class_name)]

    def generate_adversarial_example(self, tile_path):
        tile = Image.open(tile_path).convert("RGB")
        to_predict = transforms.ToTensor()(tile).unsqueeze(0)
        self.image = to_predict
        to_predict.requires_grad = True

        x_adv = to_predict.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(x_adv)

        current_alpha = self.alpha
        for step in range(self.num_steps):
            x_adv.requires_grad = True
            logits = self.torch_model(x_adv)

            original_pred = torch.argmax(self.torch_model(to_predict), dim=1)
            loss = -F.cross_entropy(logits, original_pred)
            self.torch_model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad / torch.norm(x_adv.grad, p=1)
                momentum = self.decay_factor * momentum + grad

                grad_sign = momentum.sign()
                x_adv = x_adv + current_alpha * grad_sign

                perturbation = torch.clamp(x_adv - to_predict, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(to_predict + perturbation, min=0, max=1)

                current_alpha *= self.decay_factor

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

# Parameters
# model_path = 'models/YOLO_Classification/train4/weights/best.pt'
# CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 
#          6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 
#          11: 'Stairs', 12: 'Traffic Light'}

# # Create an instance of the class and generate an adversarial example
# fgsm = ImprovedUntargetedFGSM(model_path, CLASS)
# fgsm.generate_adversarial_example("data/Training/Traffic Light/Tlight (81).png")
