# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:02:32 2025

@author: raxephion

LoRA Epoch Checker App

This script analyzes images generated at different epochs of LoRA training
by comparing them against control images (images generated with the base model
without the LoRA) and by assessing their perceptual quality.

It calculates:
- BRISQUE score for each epoch image (lower is better, indicates fewer artifacts).
- SSIM score between each epoch image and its corresponding control image
  (higher means more similar to the control, 1.0 is identical).

This helps in identifying potentially overtrained or undertrained epochs
and selecting an optimal LoRA checkpoint.

Activate and Deactivate venv (example):
# conda create -n lora_analyzer python=3.9
# conda activate lora_analyzer
# pip install -r requirements.txt
# conda deactivate
"""

import os
from pathlib import Path
from PIL import Image
# Corrected import for scikit-image < 0.16 or for newer versions where it might be aliased
# If using scikit-image >= 0.16, this might be:
# from skimage.metrics import structural_similarity as ssim
# However, your provided script uses compare_ssim, so we'll stick to that.
# Ensure scikit-image version is compatible or adjust import.
try:
    from skimage.measure import compare_ssim as ssim
except ImportError:
    # Fallback for newer scikit-image versions if compare_ssim moved
    try:
        from skimage.metrics import structural_similarity as ssim
        print("Using structural_similarity from skimage.metrics as SSIM function.")
    except ImportError:
        raise ImportError("Could not import SSIM function. Please ensure scikit-image is installed correctly.")

import numpy as np
# Corrected import based on the correct package name and internal module structure
import imquality.brisque as brisque

# --- Configuration ---
# !!! IMPORTANT !!!
# Adjust these paths to your actual folder locations.
# It's recommended to use relative paths if images are stored within the project,
# or use environment variables / configuration files for better flexibility.
LORA_EPOCH_IMAGES_DIR = Path(r"C:\...\LoRA_Training\lora_epoch_images") # Placeholder - UPDATE THIS
CONTROL_IMAGES_DIR = Path(r"C:\...\LoRA_Training\control_images") # Placeholder - UPDATE THIS
NUM_EPOCHS = 10 # Total number of epochs to check (e.g., if you trained for 10 epochs)

# Naming convention assumptions (modify if yours is different)
# Assumes epoch images are like "epoch_01.png", "epoch_02.png", ...
# And control images are like "control_01.png", "control_02.png", ...
EPOCH_IMAGE_PREFIX = "epoch_"
CONTROL_IMAGE_PREFIX = "control_"
IMAGE_EXTENSION = ".png" # Common image extension, change if different

# --- Helper Functions ---
def get_image_paths(epoch_num):
    """
    Constructs paths for epoch and control images based on epoch number.
    Adjust this function if your naming convention is different.
    """
    epoch_str = f"{epoch_num:02d}" # e.g., 1 -> "01", 10 -> "10"

    epoch_img_name = f"{EPOCH_IMAGE_PREFIX}{epoch_str}{IMAGE_EXTENSION}"
    control_img_name = f"{CONTROL_IMAGE_PREFIX}{epoch_str}{IMAGE_EXTENSION}"

    epoch_img_path = LORA_EPOCH_IMAGES_DIR / epoch_img_name
    control_img_path = CONTROL_IMAGES_DIR / control_img_name

    return epoch_img_path, control_img_path

def calculate_metrics(epoch_img_path, control_img_path):
    """
    Calculates BRISQUE for the epoch image and SSIM between epoch and control image.
    """
    try:
        epoch_img_pil = Image.open(epoch_img_path).convert('RGB') # Ensure RGB for consistency
        control_img_pil = Image.open(control_img_path).convert('RGB') # Ensure RGB for consistency

        # Convert to NumPy arrays for SSIM checks/conversion if needed
        epoch_img_np = np.array(epoch_img_pil)
        control_img_np = np.array(control_img_pil)

        # --- BRISQUE Score (for epoch image quality) ---
        # The image-quality brisque implementation takes a PIL image
        brisque_score = brisque.score(epoch_img_pil) # Pass PIL image directly


        # --- SSIM Score (similarity between epoch and control) ---
        # Ensure images are the same size for SSIM
        if epoch_img_np.shape != control_img_np.shape:
             print(f"Warning: Image shapes differ for SSIM. Epoch: {epoch_img_np.shape}, Control: {control_img_np.shape}. Resizing control to epoch size for SSIM.")
             # Resize control image to match epoch image dimensions
             control_img_pil_resized = control_img_pil.resize(epoch_img_pil.size)
             control_img_np = np.array(control_img_pil_resized) # Update numpy array
        else:
             control_img_pil_resized = control_img_pil # No resize needed

        # SSIM needs grayscale
        epoch_img_gray_np = np.array(epoch_img_pil.convert('L'))
        # Use the potentially resized control PIL image for grayscale conversion
        control_img_gray_np = np.array(control_img_pil_resized.convert('L'))


        # For scikit-image SSIM:
        # data_range is the dynamic range of the image (e.g., 255 for uint8)
        # The `full` parameter in `compare_ssim` (if used) affects the return type.
        # If `full=True`, it returns (score, gradient_image). We only need score.
        # For scikit-image < 0.16, compare_ssim(..., full=False) returns just the score.
        # For scikit-image >= 0.19, structural_similarity data_range is automatically inferred
        # if the image is a common dtype (like uint8).
        # Let's ensure data_range is explicitly set for robustness with compare_ssim
        ssim_index = ssim(epoch_img_gray_np, control_img_gray_np,
                          data_range=epoch_img_gray_np.max() - epoch_img_gray_np.min(),
                          channel_axis=None, # for grayscale, no channel axis or set to None if it causes issues
                          multichannel=False) # Explicitly state it's grayscale for older versions

        # If ssim function returns a tuple (e.g. with full=True or newer skimage versions if not handled above)
        if isinstance(ssim_index, tuple):
            ssim_index = ssim_index[0]

        return brisque_score, ssim_index

    except FileNotFoundError:
        print(f"Error: Could not find images for comparison: {epoch_img_path} or {control_img_path}")
        return None, None
    except Exception as e:
        print(f"Error processing images {epoch_img_path} / {control_img_path}: {e}")
        return None, None

# --- Main Analysis Logic ---
def analyze_epochs():
    print("Starting LoRA Epoch Analysis...")
    print(f"LoRA Epoch Images Dir: {LORA_EPOCH_IMAGES_DIR}")
    print(f"Control Images Dir: {CONTROL_IMAGES_DIR}\n")

    if not LORA_EPOCH_IMAGES_DIR.is_dir():
        print(f"Error: LoRA epoch images directory not found: {LORA_EPOCH_IMAGES_DIR}")
        print("Please update LORA_EPOCH_IMAGES_DIR in the script.")
        return
    if not CONTROL_IMAGES_DIR.is_dir():
        print(f"Error: Control images directory not found: {CONTROL_IMAGES_DIR}")
        print("Please update CONTROL_IMAGES_DIR in the script.")
        return

    # Check if directories are empty
    if not any(LORA_EPOCH_IMAGES_DIR.iterdir()):
        print(f"Error: LoRA epoch images directory is empty: {LORA_EPOCH_IMAGES_DIR}")
        return
    if not any(CONTROL_IMAGES_DIR.iterdir()):
         print(f"Error: Control images directory is empty: {CONTROL_IMAGES_DIR}")
         return

    results = []

    for i in range(1, NUM_EPOCHS + 1):
        epoch_img_path, control_img_path = get_image_paths(i)

        if not epoch_img_path.exists():
            print(f"Skipping Epoch {i}: LoRA image not found at {epoch_img_path}")
            results.append({'epoch': i, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': 'LoRA image missing'})
            continue
        if not control_img_path.exists():
            print(f"Skipping Epoch {i}: Control image not found at {control_img_path}")
            results.append({'epoch': i, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': 'Control image missing'})
            continue

        print(f"Processing Epoch {i:02d}...")
        brisque_val, ssim_val = calculate_metrics(epoch_img_path, control_img_path)

        if brisque_val is not None and ssim_val is not None:
            results.append({'epoch': i, 'brisque': brisque_val, 'ssim': ssim_val})
            print(f"  Epoch {i:02d}: BRISQUE = {brisque_val:.2f}, SSIM (to control) = {ssim_val:.4f}")
        else:
            results.append({'epoch': i, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': 'Processing error'})


    print("\n--- Analysis Summary ---")
    print("Epoch | BRISQUE (Lower is better) | SSIM (to Control, 1.0=identical)")
    print("---------------------------------------------------------------------")
    for res in results:
        if 'error' not in res:
            print(f"{res['epoch']:<5} | {res['brisque']:<25.2f} | {res['ssim']:.4f}")
        else:
            print(f"{res['epoch']:<5} | {'N/A':<25} | {'N/A'} ({res['error']})")


    # --- Suggestion for Best Epoch ---
    # Heuristic: Choose epoch with the lowest BRISQUE score among valid results.
    valid_results = [r for r in results if 'error' not in r and r['brisque'] != float('inf')]
    if not valid_results:
        print("\nNo valid results with computable metrics to suggest a best epoch.")
        return

    # Find the minimum BRISQUE score
    min_brisque_val = min(r['brisque'] for r in valid_results)

    # Find all epochs with the minimum BRISQUE score (allowing for floating point comparisons with tolerance)
    tolerance = 0.01
    best_epochs_candidates = [r for r in valid_results if abs(r['brisque'] - min_brisque_val) < tolerance]

    # If multiple epochs have the same lowest BRISQUE, choose the earliest one
    # (This is a common tie-breaking rule, you could choose the latest if preferred)
    best_epoch_info = min(best_epochs_candidates, key=lambda x: x['epoch'])


    print(f"\n--- Suggested Best Epoch (based on lowest BRISQUE) ---")
    print(f"Epoch: {best_epoch_info['epoch']}")
    print(f"  BRISQUE: {best_epoch_info['brisque']:.2f}")
    print(f"  SSIM to Control: {best_epoch_info['ssim']:.4f}")

    print("\nConsiderations for choosing the best epoch:")
    print("1. Low BRISQUE score generally indicates better perceptual image quality (fewer artifacts).")
    print("2. SSIM score indicates similarity to the control image (without LoRA).")
    print("   - If your LoRA is subtle, you might want higher SSIM.")
    print("   - If your LoRA makes significant changes, SSIM will be lower.")
    print("3. Look for a balance: good quality (low BRISQUE) and the desired level of LoRA effect (informed by SSIM).")
    print("4. Overfitting often shows as increasing BRISQUE scores in later epochs and potentially decreasing SSIM rapidly if the LoRA diverges too much from intended style or becomes noisy.")

if __name__ == "__main__":
    # Basic check for configuration paths
    if "C:\\...\\" in str(LORA_EPOCH_IMAGES_DIR) or "C:\\...\\LoRA_Training" in str(CONTROL_IMAGES_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Default placeholder paths are still in use.         !!!")
        print("!!! Please edit LORA_EPOCH_IMAGES_DIR and CONTROL_IMAGES_DIR     !!!")
        print("!!! in lora_epoch_analyzer.py before running.                    !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    
    analyze_epochs()
