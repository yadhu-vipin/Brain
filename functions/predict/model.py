import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn  
import sys
import json
import base64
from torchvision import transforms
import io
import os

# Define class labels
class_labels = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

script_dir = os.path.dirname(__file__)

# Get the absolute path of the model.pth file relative to the script directory
model_path = os.path.join(script_dir, "model.pth")

# Log the model path (this will be helpful for debugging)
print(f"Model path being used: {model_path}", file=sys.stderr)

# Load the model
def load_model():
    try:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_labels))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

model = load_model()

# Define image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

transform = get_transforms()

# Process the image and predict
def predict(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_tensor = transform(image).unsqueeze(0)

        if isinstance(model, dict):  # Return error if model failed to load
            return model

        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted_idx = torch.max(outputs, 1)

        prediction = {
            "prediction": class_labels[predicted_idx.item()],
            "class_id": predicted_idx.item(),
            "confidence": round(float(probabilities[predicted_idx.item()]) * 100, 2),
            "all_probabilities": {class_labels[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(class_labels))}
        }

        return prediction

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

if __name__ == "__main__":
    try:
        print(f"Model path being used: {model_path}", file=sys.stderr)

        # Read the base64 image data from stdin
        image_base64 = sys.stdin.read().strip()

        if not image_base64:
            raise ValueError("No image data received.")

        # Decode the base64 string
        image_data = base64.b64decode(image_base64)

        # Run prediction
        result = predict(image_data)

        # Output the result as clean JSON
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
