MODEL_PATH = "best_model.pt"
IMAGE_DIR = "./test_images"
IMAGE_EXTENSION = (
    "*.jpg"  
)

# These are derived from dataset_creation.py and sorted alphabetically
# as dictated by PyTorch's ImageFolder behavior.
CLASSES = [
    "boss-fight",
    "boss-kill",
    "choice",
    "dialog",
    "enter-level",
    "none",
    "notification",
    "victory",
]
