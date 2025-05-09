# image_test.py

import sys
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import CNN  # assumes your model class is CNN
from config import device
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(image_path):
    # Load and grayscale
    image = Image.open(image_path).convert("L")

    # Resize to 28x28 before any pixel transformation
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert to NumPy array for better pixel manipulation
    np_img = np.array(image)

    # Estimate background from corners
    corners = [np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]]
    background_mean = np.mean(corners)

    # Invert if digit is brighter than background
    if np.mean(np_img) < background_mean:
        np_img = 255 - np_img

    # Optional: Binarize (keep only 0 or 255)
    threshold = (np.max(np_img.astype(np.uint16)) + np.min(np_img.astype(np.uint16))) / 2
    np_img = np.where(np_img > threshold, 255, 0).astype(np.uint8)

    # Save debug image
    debug = Image.fromarray(np_img)
    debug.resize((280, 280), Image.NEAREST).save("debug_transformed_input.png")

    # Normalize to tensor
    tensor = transforms.ToTensor()(debug)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

    return tensor.unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_test.py path_to_image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load model
    model = CNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    digit, conf = predict(model, image_tensor)

    print(f"🧠 Predicted Digit: {digit} (Confidence: {conf * 100:.2f}%)")