# LoRA Epoch Analyzer
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script to analyze images generated at different epochs of LoRA (Low-Rank Adaptation) training. It helps in selecting an optimal LoRA checkpoint by evaluating image quality and similarity to control images.

### 🧠 1. Structural Similarity Index (SSIM)

SSIM measures the similarity between two images in terms of:

*   Luminance
*   Contrast
*   Structural features

It’s computed as:

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where:

*   $\mu_x, \mu_y$: mean pixel intensities
*   $\sigma_x, \sigma_y$: standard deviations
*   $\sigma_{xy}$: cross-covariance
*   $C_1, C_2$: constants to stabilize division

**Interpretation:**

*   SSIM ≈ 1.0 → Very similar (minimal LoRA effect)
*   SSIM ≪ 1.0 → Significant difference (strong LoRA effect)

### 🔍 2. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

BRISQUE estimates the perceptual quality of an image without needing a reference. It uses machine learning and natural scene statistics to assess artifacts and distortions.

**How it works:**

*   Extracts statistical features from image patches
*   Feeds them into a pretrained model (typically SVM)
*   Outputs a quality score

**Interpretation:**

*   Lower score → Better image quality
*   Higher score → More visible artifacts, noise, or degradation

## 🎯 Goal

By analyzing both:

*   **SSIM** (similarity to original)
*   **BRISQUE** (perceptual quality)

The tool helps you:

*   Detect the best LoRA epoch for subtle or strong stylistic changes
*   Avoid over-strengthening that introduces artifacts
*   Maintain good image quality while applying desired effects


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
    git clone https://github.com/Raxephion/loRA-Epoch-Analyser.git
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
    CONTROL_IMAGES_DIR = Path(r"C:\...\LoRA_Training\control_images") # UPDATE THIS
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
