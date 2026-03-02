# main.py
import os
import json
from datetime import datetime

from config import CLASSES, IMAGE_DIR, IMAGE_EXTENSION, MODEL_PATH
from file_handler import get_sorted_image_paths
from inference import PyTorchInferencer
from event_interval_detector import segment_events


def save_classifications_to_file(classified_images):
    """
    Saves the classified image labels to a JSON file
    in the current working directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classified_run_{timestamp}.json"
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(classified_images, f, indent=2)

    print(f"\nSaved classifications to: {filepath}")

def save_final_results_to_file(final_results):
    """
    Saves the final segmented event intervals to a JSON file
    in the current working directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_results_{timestamp}.json"
    filepath = os.path.join(os.getcwd(), filename)

    # Convert EventInterval objects to dicts for JSON serialization
    results_to_save = [
        {"label": iv.label, "start": iv.start, "end": iv.end, "length": iv.length}
        for iv in final_results
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nSaved final results to: {filepath}")

def main():
    image_paths = get_sorted_image_paths(IMAGE_DIR, IMAGE_EXTENSION)
    if not image_paths:
        print("No images found.")
        return

    inferencer = PyTorchInferencer(MODEL_PATH, CLASSES)

    classified_images = []
    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        image_result = inferencer.process_image(img_path)
        classified_images.append(image_result)

    # ---- Save array BEFORE segmentation ----
    save_classifications_to_file(classified_images)

    # ---- Segment events ----
    final_results = segment_events(classified_images)

    print("\n--- Final Results ---")
    for iv in final_results:
        print(iv.label, iv.start, iv.end, iv.length)
    
    save_final_results_to_file(final_results)


if __name__ == "__main__":
    main()