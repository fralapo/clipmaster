import torch
import json
import os

API_KEYS_FILE = "api_keys.json"

def save_api_keys(keys_dict):
    """Saves the API keys to a JSON file."""
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(keys_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving API keys: {e}")
        return False

def load_api_keys():
    """Loads the API keys from a JSON file."""
    if not os.path.exists(API_KEYS_FILE):
        return {}
    try:
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return {}

def get_recommended_whisper_model():
    """
    Recommends a Whisper model based on available VRAM.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Recommending 'base' model for CPU.")
        return "base"

    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # VRAM in GB
    print(f"Available VRAM: {vram:.2f} GB")

    if vram >= 10:
        return "large"
    elif vram >= 5:
        return "medium"
    elif vram >= 2:
        return "small"
    else:
        return "base"

import re

def cleanup_temp_files(temp_dir="temp"):
    """
    Deletes all files in the temporary directory.
    """
    import os
    import shutil
    if os.path.exists(temp_dir):
        print(f"Cleaning up temporary files in {temp_dir}...")
        shutil.rmtree(temp_dir)
        print("Cleanup complete.")

def sanitize_filename(filename):
    """
    Sanitizes a string to be a valid filename, preserving spaces.
    """
    # Remove illegal characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Truncate to a reasonable length
    return sanitized[:100]
