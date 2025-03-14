# api/predict.py
from http.server import BaseHTTPRequestHandler
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import sys

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)
# Add project root to path to help with imports
sys.path.append(project_root)

# Load model setup (this will run on cold start)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)

# Load model from the relative path in the project
model_path = os.path.join(project_root, "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def parse_multipart(body_bytes, content_type):
    # Simple parser for multipart/form-data
    boundary = content_type.split('boundary=')[1].encode()
    
    parts = body_bytes.split(boundary)
    for part in parts:
        if b'filename=' in part and b'name="file"' in part:
            # Find the file data
            headers_end = part.find(b'\r\n\r\n')
            if headers_end > 0:
                file_data = part[headers_end+4:]
                if b'\r\n--' in file_data:
                    file_data = file_data[:file_data.rfind(b'\r\n--')]
                return file_data
    return None

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Get content length and type
        content_length = int(self.headers['Content-Length'])
        content_type = self.headers['Content-Type']
        
        # Read the POST body
        body = self.rfile.read(content_length)
        
        try:
            # Parse the multipart form data
            if 'multipart/form-data' in content_type:
                file_data = parse_multipart(body, content_type)
                if not file_data:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No file provided"}).encode())
                    return
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Content-Type must be multipart/form-data"}).encode())
                return
            
            # Process image
            image = Image.open(io.BytesIO(file_data))
            
            # Apply transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run prediction
            with torch.no_grad():
                output = model(img_tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            tumor_types = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
            result = tumor_types[prediction]
            
            # Return result
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"prediction": result}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Prediction failed: {str(e)}"}).encode())

    def do_GET(self):
        # Simple health check
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"message": "Brain Tumor Classification API is live!"}).encode())