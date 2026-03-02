# file_handler.py
import glob
import os


def extract_serial_number(filepath):
    """
    Helper function to extract the integer serial number from a path.
    Expects format: /path/to/file/<filename>_<serial_number>.<ext>
    """
    # 1. Isolate just the filename (e.g., "game_screen_042.jpg")
    base_name = os.path.basename(filepath)

    # 2. Strip off the extension (e.g., "game_screen_042")
    name_without_ext = os.path.splitext(base_name)[0]

    try:
        # 3. Split from the right side at the first underscore it finds,
        #    grab the last piece, and convert to integer.
        serial_str = name_without_ext.rsplit("_x", 1)[-1]
        return int(serial_str)
    except (ValueError, IndexError):
        # Failsafe: If a file doesn't have an underscore or number,
        # send it to the very beginning of the list.
        return -1


def get_sorted_image_paths(image_dir, extension="*.jpg"):
    """
    Scans a directory for images and returns them sorted by their numeric serial number.
    """
    search_pattern = os.path.join(image_dir, extension)

    # Get all matching files
    image_paths = glob.glob(search_pattern)

    # Sort the files using our custom integer extraction function
    image_paths.sort(key=extract_serial_number)

    if not image_paths:
        print(f"Warning: No images found in '{image_dir}' matching '{extension}'")

    return image_paths
