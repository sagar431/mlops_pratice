# Image Classification API

A FastAPI-based image classification service using TIMM (PyTorch Image Models) with a MobileNetV2 architecture.

## Features

- Image classification using the MobileNetV2 model trained on ImageNet
- RESTful API for easy integration
- Accepts base64-encoded images
- Returns top 5 predictions with probabilities
- GPU acceleration support

## Project Structure

```
.
├── src/
│   ├── server.py         # FastAPI server implementation
│   └── test.client.py    # Test client for the API
└── README.md
```

## Usage

### Starting the Server

```bash
python src/server.py
```

The server will start on http://0.0.0.0:8000

### Making Predictions

Use the test client to send image requests:

```bash
python src/test.client.py
```

Or make a POST request with base64-encoded image:

```python
import requests
import base64

# Encode image as base64
with open("image.jpg", "rb") as f:
    img_bytes = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"image": img_bytes}
)

# Process response
if response.status_code == 200:
    predictions = response.json()["predictions"]
    for pred in predictions:
        print(f"{pred['label']}: {pred['probability']:.2%}")
```

## API Documentation

API documentation is available at http://127.0.0.1:8000/docs when the server is running. 