import requests
from urllib.request import urlopen
import base64

def test_single_image():
    # Get test image
    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    img_data = urlopen(url).read()
    
    # Convert to base64 string
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    
    # Debug: Print some information about the image
    print(f"Image data size: {len(img_data)} bytes")
    print(f"Base64 encoded size: {len(img_bytes)} characters")
    
    try:
        # Send request - try with 127.0.0.1 instead of localhost
        print("Sending request to server...")
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"image": img_bytes},  # Send as JSON instead of files
            timeout=30  # Add timeout
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            print("\nTop 5 Predictions:")
            for pred in predictions:
                print(f"{pred['label']}: {pred['probability']:.2%}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception during request: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_single_image()
