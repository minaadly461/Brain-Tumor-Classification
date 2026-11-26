
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_resnet50.pth"

# Import project modules
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data import CLASS_NAMES, DEVICE
from resnet_model import ResNet50Classifier


# Inference configuration
IMAGE_SIZE = 256

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BrainTumorPredictor:

    def __init__(self, model_path=MODEL_PATH):

        self.device = DEVICE

        # Load model architecture
        self.model = ResNet50Classifier(
            num_classes=len(CLASS_NAMES),
            pretrained=False
        )

        # Load trained weights
        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Inference preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        ])

    def predict(self, image_input):

        # Accept either an image path or a PIL Image
        if isinstance(image_input, (str, Path)):

            image_path = Path(image_input)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            with Image.open(image_path) as img:
                image = img.convert("RGB")

        elif isinstance(image_input, Image.Image):

            image = image_input.convert("RGB")

        else:
            raise TypeError(
                "Input must be an image path or PIL Image."
            )

        # Preprocess image
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Model inference
        with torch.inference_mode():

            outputs = self.model(image_tensor)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

        probabilities = probabilities.squeeze(0).cpu()

        predicted_idx = probabilities.argmax().item()

        # Prediction results
        results = {
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": round(
                probabilities[predicted_idx].item(), 4
            ),
            "probabilities": {
                class_name: round(prob.item(), 4)
                for class_name, prob in zip(
                    CLASS_NAMES,
                    probabilities
                )
            }
        }

        return results
