import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from flask import Flask, request, jsonify
from utils.network_weight import UNet
from utils.network import UNet as HUNet
from utils.bmi_calculator import BMI_calculator

app = Flask(__name__)

# Initialize your Azure Blob Storage client
connection_string = "DefaultEndpointsProtocol=https;AccountName=modelosemillero;AccountKey=tyBROI5tn6oxVpk1K/nKL8muX9ZKy8OsMHplkLtyyyNDC4vG3QSxWw22uIDiMthDuYWNSEX006un+AStCP3YHw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "semillero"
model_height_blob = "model_ep_48.pth.tar"
model_weight_blob = "model_ep_37.pth.tar"

# Define a function to load models from Azure Blob Storage
def load_models_from_blob():
    model_h_blob_client = blob_service_client.get_blob_client(container_name, model_height_blob)
    model_w_blob_client = blob_service_client.get_blob_client(container_name, model_weight_blob)

    # Load the height model
    height_model_bytes = model_h_blob_client.download_blob().readall()
    height_model_stream = BytesIO(height_model_bytes)
    model_h = HUNet(128)
    model_h.load_state_dict(torch.load(height_model_stream, map_location=torch.device('cpu'))["state_dict"])

    # Load the weight model
    weight_model_bytes = model_w_blob_client.download_blob().readall()
    weight_model_stream = BytesIO(weight_model_bytes)
    model_w = UNet(128, 32, 32)
    model_w.load_state_dict(torch.load(weight_model_stream, map_location=torch.device('cpu'))["state_dict"])

    if torch.cuda.is_available():
        model_w = model_w.cuda(3)
        model_h = model_h.cuda(3)

    return model_h, model_w

# Load models from Azure Blob Storage
model_h, model_w = load_models_from_blob()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Rest of your prediction code remains the same

        # Get the uploaded image from the POST request
        image_data = request.files['image']

        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        RES = 128
        scale = RES / max(image.shape[:2])

        X_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if X_scaled.shape[1] > X_scaled.shape[0]:
            p_a = (RES - X_scaled.shape[0]) // 2
            p_b = (RES - X_scaled.shape[0]) - p_a
            X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0, 0)], mode='constant')
        elif X_scaled.shape[1] <= X_scaled.shape[0]:
            p_a = (RES - X_scaled.shape[1]) // 2
            p_b = (RES - X_scaled.shape[1]) - p_a
            X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0, 0)], mode='constant')

        X = (X * 255).astype('uint8')
        X = transforms.ToTensor()(X).unsqueeze(0)

        if torch.cuda.is_available():
            X = X.cuda()

        model_w.eval()
        with torch.no_grad():
            m_p, j_p, _, w_p = model_w(X)

        model_h.eval()
        with torch.no_grad():
            _, _, h_p = model_h(X)

        height_cm = 100 * h_p.item()
        weight_kg = 100 * w_p.item()
        BMI = weight_kg / ((height_cm) / 100) ** 2
        bmi_result = BMI_calculator(BMI)

        # Create a JSON response
        response = {
            "altura_cm": height_cm,
            "peso_kg": weight_kg,
            "BMI": BMI,
            "estado": bmi_result
        }

        return jsonify(response)

    except Exception as e:
        return str(e), 400  # Return an error message if something goes wrong

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
