from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Random colors for visualization
COLORS = np.random.uniform(0, 255, size=(80, 3))


def parse_detections(result):
    """Extract top classification result."""
    probabilities = result.probs.data.cpu().numpy()
    top_class = np.argmax(probabilities)
    confidence = probabilities[top_class]
    class_name = result.names[top_class]
    return top_class, class_name, confidence


def draw_detections(top_class, class_name, confidence, img):
    """Draw predicted class and confidence on the image."""
    label = f"{class_name} ({confidence:.2f})"
    color = COLORS[top_class]
    cv2.putText(
        img, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA
    )
    return img


def main():
    # Load and preprocess the image
    image_url = "../data/Training/Motorcycle/Motorcycle (75).png"
    img = Image.open(image_url).convert("RGB")  # Ensure 3 channels (RGB)
    img = np.array(img)
    img = cv2.resize(img, (640, 640))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    # Load the YOLO model
    model_path = '../models/YOLO_Classification/train4/weights/best.pt'
    model = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)
    print(model.model.model)
    # Select the last convolutional layer for Grad-CAM
    target_layers = [model.model.model[-2]]

    # Run the model
    results = model([rgb_img])
    result = results[0]

    # Parse detections
    top_class, class_name, confidence = parse_detections(result)
    print(f"Predicted class: {class_name}, Confidence: {confidence}")

    # Draw detections on the original image
    img_with_label = draw_detections(top_class, class_name, confidence, rgb_img.copy())

    # Grad-CAM visualization
    tensor = tensor.to(device)
    cam = EigenCAM(model=model.model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # Display the original image with label and Grad-CAM visualization
    Image.fromarray((visualization * 255).astype(np.uint8)).show()
    Image.fromarray(img_with_label).show()


if __name__ == "__main__":
    main()
