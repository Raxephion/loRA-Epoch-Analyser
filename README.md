# LoRA Epoch Analyzer

A Python script to analyze images generated at different epochs of LoRA (Low-Rank Adaptation) training. It helps in selecting an optimal LoRA checkpoint by evaluating image quality and similarity to control images.

## Features

-   Calculates **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)** score for each epoch image. Lower BRISQUE scores generally indicate better perceptual quality (fewer artifacts).
-   Calculates **SSIM (Structural Similarity Index Measure)** between each epoch image and its corresponding control image (generated with the base model without LoRA). An SSIM score of 1.0 means the images are identical.
-   Provides a summary table of scores for all epochs.
-   Suggests a "best" epoch based on the lowest BRISQUE score (with earliest epoch as a tie-breaker).

## Prerequisites

-   Python 3.7+
-   Git (for cloning)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd lora-epoch-analyzer
    ```

2.  **Create and activate a virtual environment:**

    *   Using `venv`:
        ```bash
        python -m venv venv
        # On Windows
        .\venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n lora_analyzer python=3.9 # Or your preferred Python 3.x version
        conda activate lora_analyzer
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Paths:**
    Open `lora_epoch_analyzer.py` in a text editor.
    You **MUST** update the following placeholder paths to your actual directory locations:
    ```python
    LORA_EPOCH_IMAGES_DIR = Path(r"C:\...\LoRA_Training\lora_epoch_images") # UPDATE THIS
    CONTROL_IMAGES_DIR = Path(r"C:\Users\raxep\OneDrive\Desktop\LoRA_Training\control_images") # UPDATE THIS
    ```
    Also, adjust `NUM_EPOCHS` if needed:
    ```python
    NUM_EPOCHS = 10 # Total number of epochs trained and to be checked
    ```

5.  **Verify Image Naming Convention:**
    The script assumes your images are named as follows:
    -   Epoch images: `epoch_01.png`, `epoch_02.png`, ..., `epoch_10.png`
    -   Control images: `control_01.png`, `control_02.png`, ..., `control_10.png`

    If your naming convention is different, modify these variables in `lora_epoch_analyzer.py`:
    ```python
    EPOCH_IMAGE_PREFIX = "epoch_"
    CONTROL_IMAGE_PREFIX = "control_"
    IMAGE_EXTENSION = ".png"
    ```
    The script expects one control image for each epoch image, corresponding to the base model's output for the same prompt/seed used to generate that epoch's LoRA image.

## Usage

Once configured, run the script from your terminal:

```bash
python lora_epoch_analyzer.py
