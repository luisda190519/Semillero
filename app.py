import os
import time
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from network_weight import UNet
from network import UNet as HUNet
from flask import Flask, request, jsonify
from draw_skeleton import create_colors, draw_skeleton

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the POST request
        image_data = request.files['image']

        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

        # Load the height and weight models
        model_h = HUNet(128)
        pretrained_model_h = torch.load('models/model_ep_48.pth.tar', map_location=torch.device('cpu'))
        model_h.load_state_dict(pretrained_model_h["state_dict"])

        model_w = UNet(128, 32, 32)
        pretrained_model_w = torch.load('models/model_ep_37.pth.tar', map_location=torch.device('cpu'))
        model_w.load_state_dict(pretrained_model_w["state_dict"])

        if torch.cuda.is_available():
            model = model_w.cuda(3)
        else:
            model = model_w

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

        model.eval()
        with torch.no_grad():
            m_p, j_p, _, w_p = model(X)

        del model

        if torch.cuda.is_available():
            model = model_h.cuda(3)
        else:
            model = model_h

        model.eval()
        with torch.no_grad():
            _, _, h_p = model(X)

        # ... Continue with the image preprocessing code as in the original script

        # Convert height and weight predictions to human-readable values
        height_cm = 100 * h_p.item()
        weight_kg = 100 * w_p.item()

        # Create a JSON response
        response = {
            "height_cm": height_cm,
            "weight_kg": weight_kg
        }

        return jsonify(response)

    except Exception as e:
        return str(e), 400  # Return an error message if something goes wrong


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
