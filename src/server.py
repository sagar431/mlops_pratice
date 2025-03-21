import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        logger.info(f"Setting up model on device: {device}")
        self.device = device
        # Create model and move to appropriate device
        try:
            self.model = timm.create_model('mambaout_base.in1k', pretrained=True)
            self.model = self.model.to(device)
            self.model.eval()
            logger.info("Model loaded successfully")

            # Get model specific transforms
            data_config = timm.data.resolve_model_data_config(self.model)
            self.transforms = timm.data.create_transform(**data_config, is_training=False)
            logger.info("Transforms created successfully")

            # Load ImageNet labels
            import requests
            url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
            self.labels = requests.get(url).text.strip().split('\n')
            logger.info(f"Loaded {len(self.labels)} class labels")
        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        try:
            logger.info(f"Received request with keys: {request.keys()}")
            image_bytes = request.get("image")
            if not image_bytes:
                logger.error("No image data provided")
                raise ValueError("No image data provided")
            
            logger.info(f"Image data received, length: {len(image_bytes)}")
            
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image_bytes)
            logger.info(f"Decoded base64 data, length: {len(img_bytes)}")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            logger.info(f"Converted to PIL Image: {image.format}, {image.size}")
            
            # Convert to tensor and move to device
            tensor = self.transforms(image).unsqueeze(0).to(self.device)
            logger.info(f"Image transformed to tensor: {tensor.shape}")
            return tensor
        except Exception as e:
            logger.error(f"Error in decode_request: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def predict(self, x):
        try:
            logger.info(f"Running prediction on tensor of shape: {x.shape}")
            outputs = self.model(x)
            logger.info(f"Model output shape: {outputs.shape}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            logger.info(f"Probabilities calculated")
            return probabilities
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def encode_response(self, output):
        """Convert model output to API response"""
        try:
            logger.info(f"Encoding response from output of shape: {output.shape}")
            # Get top 5 predictions
            probs, indices = torch.topk(output[0], k=5)
            
            predictions = [
                {
                    "label": self.labels[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
            
            logger.info(f"Top prediction: {predictions[0]['label']} ({predictions[0]['probability']:.4f})")
            
            return {
                "predictions": predictions
            }
        except Exception as e:
            logger.error(f"Error in encode_response: {str(e)}")
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    api = ImageClassifierAPI()
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
    )
    server.run(port=8000)