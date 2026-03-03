import io

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class PyTorchInferencer:
    def __init__(self, model_path, class_names):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading timm ConvNeXt-Tiny on {self.device}...")

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,  # False because we are loading custom weights
            num_classes=len(self.class_names),
        )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        img_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def process_image(self, img_bytes: bytes, capture_id: str):
        try:
            image_stream = io.BytesIO(img_bytes)
            image = Image.open(image_stream).convert("RGB")
            input_tensor = self.transform(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_batch)

            probabilities = F.softmax(output[0], dim=0)
            confidence, class_id = torch.max(probabilities, 0)

            class_name = self.class_names[class_id.item()]

            return class_name

        except Exception as e:
            print(f"Error processing capture_id: {capture_id}: {e}")
            return {
                "capture_id": capture_id,
                "class_name": "Error",
                "classification_confidence": 0.0,
            }
