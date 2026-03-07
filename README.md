

```markdown
# Drone Image Quality Assessment & Self-Learning Pipeline

## Project Overview

Drone-based photogrammetry and 3D reconstruction pipelines require **high-quality images**. Poor-quality images (blurred, overexposed, noisy, or textureless images) can significantly degrade the quality of the generated 3D models.

This project develops an **intelligent multi-stage pipeline** that automatically evaluates drone images and classifies them into:

- **GOOD** → Suitable for direct use in 3D reconstruction
- **RECOVERABLE** → Can potentially be enhanced and reused
- **BAD** → Should be discarded

Instead of performing only image classification, this system builds a **complete adaptive workflow** that:

1. Extracts image quality metrics
2. Trains a CNN classifier (EfficientNet-B0)
3. Enhances recoverable images
4. Re-evaluates image quality
5. Logs results for monitoring
6. Supports a **self-learning pipeline**

The goal is to improve the quality of input images before they are used in **3D mapping systems such as COLMAP or OpenSfM**.

---

# Dataset

Due to GitHub size limitations, all datasets are hosted on Google Drive.

## Raw Dataset (Initial Image Collection)

Initial drone image datasets used for labeling and feature extraction:

Raw dataset folders:

https://drive.google.com/drive/folders/1WxPWBNJL9MHO-BT16NifqncJGqisxGRE?usp=drive_link

https://drive.google.com/drive/folders/1FO_usOJZJzLATcPgPTbnAxgQRZmT6r6j?usp=drive_link


These datasets contain mixed drone images that were manually inspected and categorized during the early stages of the project.

---

# Feature Extraction

Before training the deep learning model, several **image quality metrics** were extracted from the images.

These metrics help quantify image quality characteristics such as blur, brightness, texture richness, and atmospheric noise.

Metrics extracted include:

- Laplacian variance (blur detection)
- ORB keypoints (texture richness)
- Brightness
- Saturation
- Haze estimation

Feature extraction results were stored as CSV files.

Feature datasets:

Good images feature dataset:
https://drive.google.com/file/d/1Sti3eoVucljoF2zp3rV7jf4uOI4tPySq/view?usp=sharing

Bad images feature dataset:
https://drive.google.com/file/d/1WcS9483Trvn5NIMq9utH3RMRbbUZyWPm/view?usp=sharing


Using these extracted features, the images were labeled into:

- GOOD
- BAD
- RECOVERABLE

After labeling, the images were placed into their respective folders according to their class.

---

# Data Augmentation

To increase dataset size and improve model robustness, augmentation techniques were applied to the labeled images.

Augmentation included:

- horizontal flipping
- brightness variation
- contrast variation
- minor rotations

Augmented datasets:

Augmented BAD images:
https://drive.google.com/drive/folders/12YavNN2qXg_eIDFUCdLprkzWKApGThTb?usp=sharing

Augmented GOOD images:
https://drive.google.com/drive/folders/1jRV76EtcYttc3d-ozkxyWH3B8f2S3R2o?usp=sharing

These augmented datasets helped balance the dataset and improve CNN training.

---

# Training Dataset Split

After augmentation, the dataset was split into:

- **Training set**
- **Validation set**
- **Testing set**

This structured dataset was used to train the EfficientNet model.

Dataset split structure:

```

dataset/
train/
GOOD/
BAD/
RECOVERABLE/

```
val/
    GOOD/
    BAD/
    RECOVERABLE/

test/
    GOOD/
    BAD/
    RECOVERABLE/
```

```

---

# Model Training

The classification model used in this project is:

**EfficientNet-B0**

EfficientNet was chosen because it provides:

- strong performance
- efficient parameter usage
- good generalization on image classification tasks

Training pipeline included:

1. Loading EfficientNet-B0
2. Replacing the final classification layer
3. Training on the augmented dataset
4. Validating performance on the validation set
5. Evaluating final performance on the test dataset

---

# Main Project Pipeline

The core contribution of this project is a **multi-stage intelligent pipeline** that processes drone images before they enter the photogrammetry workflow.

<img width="520" height="538" alt="image" src="https://github.com/user-attachments/assets/8b233cf7-2de8-4b9c-98cc-a647ca67eb02" />


---

# Image Enhancement

If an image is classified as **RECOVERABLE**, the system attempts to improve it using enhancement techniques such as:

- contrast enhancement
- sharpening
- noise reduction

The enhanced image is then **re-classified** by the CNN.

If the enhanced image is predicted as **GOOD**, it is accepted into the usable dataset.

---

# Self-Learning Mechanism

The pipeline supports **continual learning**.

If the model predicts an image with **high confidence**, that image can be added to the training dataset for future retraining.

This allows the system to gradually improve its performance as more drone images are processed.

---
<img width="357" height="586" alt="image" src="https://github.com/user-attachments/assets/b15997a3-2d15-431c-b35e-526697e3fccc" />


---

# Running the Pipeline

Step 1 — Clone the repository

```

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)

```

Step 2 — Install dependencies

```

pip install torch torchvision opencv-python pandas scikit-learn pillow

```

Step 3 — Add new drone images

Place images in:

```

data/incoming_drone_images

```

Step 4 — Run the pipeline

```

python pipeline/run_full_pipeline.py

```

The system will:

1. extract quality metrics
2. classify images
3. enhance recoverable images
4. re-classify enhanced images
5. save results

---

# Example Output

```

Processing: image1.jpg
Processing: image2.jpg
Processing: image3.jpg

Pipeline finished.
Results saved to data/processed/pipeline_results.csv

```

---

# Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Pandas
- Scikit-learn

---

# Future Improvements

Possible future improvements include:

- GAN-based image restoration
- integration with COLMAP to evaluate reconstruction quality
- automated threshold tuning
- improved continual learning strategies

---


