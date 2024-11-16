from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model_path = '../models/YOLO_Classification/train4/weights/best.pt'
model = YOLO(model_path)

# Grad-CAM implementation for YOLO
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Hook to capture gradients and feature maps
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_index=None):
        input_tensor.requires_grad = True  # Enable gradient tracking

        # Perform forward pass
        self.model.model.eval()  # Ensure evaluation mode
        logits = self.model.model(input_tensor)

        # Use the prediction class if no class index is specified
        if class_index is None:
            class_index = logits.argmax().item()

        # Compute the loss for the target class
        target_score = logits[:, class_index].sum()
        self.model.model.zero_grad()  # Clear gradients
        target_score.backward(retain_graph=True)  # Backpropagate

        # Get gradients and feature maps
        gradients = self.gradients  # Gradients of the target layer
        feature_maps = self.feature_maps  # Output of the target layer

        # Debug: Print gradient and feature map shapes
        print(f"Gradients Shape: {gradients.shape}")
        print(f"Feature Maps Shape: {feature_maps.shape}")

        # Compute weights for each channel by spatially averaging gradients
        if gradients.dim() == 4:  # Ensure gradients have spatial dimensions
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Gradients lack spatial dimensions for Grad-CAM.")

        # Generate the weighted combination of feature maps
        cam = torch.sum(weights * feature_maps, dim=1).squeeze(0)  # Weighted sum across channels

        # Apply ReLU to retain positive activations only
        cam = torch.relu(cam)

        # Normalize the heatmap to range [0, 1]
        cam -= cam.min()
        cam /= cam.max()

        return cam

def show_heatmap_on_image(input_image, heatmap, output_path=None):
    # Ensure the heatmap is converted to a NumPy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Normalize heatmap to range [0, 255] and convert to uint8
    heatmap = (heatmap * 255 / np.max(heatmap)).astype(np.uint8)  # Normalize and convert to uint8
    heatmap = cv2.resize(heatmap, (input_image.width, input_image.height))  # Resize to match input image dimensions

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on the input image
    input_image_np = np.array(input_image)
    overlayed_image = cv2.addWeighted(cv2.cvtColor(input_image_np, cv2.COLOR_RGB2BGR), 0.5, heatmap_color, 0.5, 0)

    # Display the heat map
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Save the image if a path is provided
    if output_path:
        cv2.imwrite(output_path, overlayed_image)

def predict_with_heatmap(tile_path):
    tile = Image.open(tile_path).convert("RGB")

    # Resize the image to 256x256 as required
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.ToTensor()          # Convert to tensor
    ])
    to_predict = preprocess(tile).unsqueeze(0)  # Shape: (1, 3, 256, 256)

    # Perform prediction
    results = model(to_predict)
    result = results[0]

    # Print the prediction
    class_index = result.probs.top1  # Class with the highest confidence
    class_name = result.names[class_index]
    confidence = result.probs.top1conf
    print(f"Prediction: {class_name} with confidence {confidence:.4f}")

    # Initialize Grad-CAM with correct convolutional layer
    target_layer = model.model.model[-2]  # Adjust based on model architecture
    grad_cam = GradCAM(model, target_layer)

    # Generate heat map for the top prediction
    heatmap = grad_cam.generate_heatmap(to_predict, class_index=class_index)

    # Visualize the heat map on the 256x256 input image
    show_heatmap_on_image(tile.resize((256, 256)), heatmap, output_path="heatmap_overlay.png")

# Test the function
predict_with_heatmap("../data/Training/Bicycle/Bicycle (75).png")
